from __future__ import annotations

import os

BOXING_TF_DEVICE = os.environ.get("BOXING_TF_DEVICE", "cpu").strip().lower()
if BOXING_TF_DEVICE == "cpu":
  os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback

from boxing import BoxingConfig, evaluate
from distill_boxing import DistillationConfig, train_student
from expert.cleanrl_boxing_teacher import DEFAULT_FILENAME, collect_trajectories, load_teacher


def now_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ContinuousTrainingConfig:
  teacher_model: Path = Path("artifacts/teacher_cleanrl") / DEFAULT_FILENAME
  dataset_dir: Path = Path("artifacts/teacher_boxing_dataset_continuous")
  student_dir: Path = Path("artifacts/student_imitation_continuous")
  evaluation_dir: Path = Path("artifacts/student_imitation_continuous_eval")
  control_dir: Path = Path("artifacts/continuous_training")
  teacher_bootstrap_episodes: int = 300
  teacher_episodes_per_collection: int = 100
  collect_after_no_improvement_rounds: int = 2
  teacher_shard_size: int = 2048
  student_epochs_per_round: int = 3
  student_batch_size: int = 128
  student_learning_rate: float = 3e-4
  student_validation_fraction: float = 0.1
  student_shuffle_buffer: int = 4096
  lr_decay_patience_rounds: int = 4
  learning_rate_decay: float = 0.5
  min_learning_rate: float = 1e-5
  evaluation_episodes: int = 10
  seed: int = 42
  retry_delay_seconds: int = 60
  idle_seconds: int = 5


def state_path(config: ContinuousTrainingConfig) -> Path:
  return config.control_dir / "run_state.json"


def events_path(config: ContinuousTrainingConfig) -> Path:
  return config.control_dir / "events.jsonl"


def stop_file_path(config: ContinuousTrainingConfig) -> Path:
  return config.control_dir / "STOP_CONTINUOUS_TRAINING"


def best_env_model_path(config: ContinuousTrainingConfig) -> Path:
  return config.student_dir / "best_env_model.h5"


def best_env_eval_path(config: ContinuousTrainingConfig) -> Path:
  return config.student_dir / "best_env_evaluation.json"


def ensure_directories(config: ContinuousTrainingConfig) -> None:
  config.dataset_dir.mkdir(parents=True, exist_ok=True)
  config.student_dir.mkdir(parents=True, exist_ok=True)
  config.evaluation_dir.mkdir(parents=True, exist_ok=True)
  config.control_dir.mkdir(parents=True, exist_ok=True)


def append_event(config: ContinuousTrainingConfig, event: str, payload: dict) -> None:
  ensure_directories(config)
  record = {
    "timestamp": now_iso(),
    "event": event,
    "payload": payload,
  }
  with events_path(config).open("a", encoding="utf-8") as file:
    file.write(json.dumps(record) + "\n")
  print(json.dumps(record), flush=True)


def default_state(config: ContinuousTrainingConfig) -> dict:
  return {
    "config": {
      **asdict(config),
      "teacher_model": str(config.teacher_model),
      "dataset_dir": str(config.dataset_dir),
      "student_dir": str(config.student_dir),
      "evaluation_dir": str(config.evaluation_dir),
      "control_dir": str(config.control_dir),
    },
    "status": "initializing",
    "active_phase": "idle",
    "round_index": 0,
    "last_completed_round": 0,
    "current_round": None,
    "total_teacher_episodes": 0,
    "total_teacher_samples": 0,
    "last_collect_round": 0,
    "last_collection_trigger_no_improvement_count": None,
    "total_student_rounds": 0,
    "total_student_epochs": 0,
    "learning_rate": config.student_learning_rate,
    "rounds_since_improvement": 0,
    "last_lr_decay_trigger_no_improvement_count": None,
    "best_eval_reward": None,
    "best_eval_model_path": None,
    "best_eval_summary_path": None,
    "latest_model_path": None,
    "failure_count": 0,
    "last_error": None,
    "updated_at": now_iso(),
  }


def load_state(config: ContinuousTrainingConfig) -> dict:
  path = state_path(config)
  if not path.exists():
    return default_state(config)

  with path.open("r", encoding="utf-8") as file:
    state = json.load(file)

  defaults = default_state(config)
  defaults.update(state)
  defaults["config"] = defaults["config"]
  return defaults


def save_state(config: ContinuousTrainingConfig, state: dict) -> None:
  ensure_directories(config)
  state["updated_at"] = now_iso()
  with state_path(config).open("w", encoding="utf-8") as file:
    json.dump(state, file, indent=2)


def stop_requested(config: ContinuousTrainingConfig) -> bool:
  return stop_file_path(config).exists()


def load_dataset_manifest(dataset_dir: Path) -> dict:
  manifest_path = dataset_dir / "manifest.json"
  if not manifest_path.exists():
    return {}

  with manifest_path.open("r", encoding="utf-8") as file:
    return json.load(file)


def sync_dataset_state(config: ContinuousTrainingConfig, state: dict) -> None:
  manifest = load_dataset_manifest(config.dataset_dir)
  state["total_teacher_episodes"] = int(manifest.get("num_episodes", 0))
  state["total_teacher_samples"] = int(manifest.get("num_samples", 0))


def resolve_collection_increment(config: ContinuousTrainingConfig, state: dict) -> int:
  total_teacher_episodes = int(state.get("total_teacher_episodes", 0))
  if total_teacher_episodes < config.teacher_bootstrap_episodes:
    return config.teacher_bootstrap_episodes - total_teacher_episodes

  rounds_since_improvement = int(state.get("rounds_since_improvement", 0))
  if rounds_since_improvement <= 0:
    return 0

  if rounds_since_improvement % config.collect_after_no_improvement_rounds != 0:
    return 0

  last_trigger = state.get("last_collection_trigger_no_improvement_count")
  if last_trigger == rounds_since_improvement:
    return 0

  return config.teacher_episodes_per_collection


def maybe_decay_learning_rate(config: ContinuousTrainingConfig, state: dict) -> dict | None:
  rounds_since_improvement = int(state.get("rounds_since_improvement", 0))
  if rounds_since_improvement <= 0:
    return None

  if rounds_since_improvement % config.lr_decay_patience_rounds != 0:
    return None

  last_trigger = state.get("last_lr_decay_trigger_no_improvement_count")
  if last_trigger == rounds_since_improvement:
    return None

  previous_lr = float(state.get("learning_rate", config.student_learning_rate))
  updated_lr = max(previous_lr * config.learning_rate_decay, config.min_learning_rate)
  state["last_lr_decay_trigger_no_improvement_count"] = rounds_since_improvement
  state["learning_rate"] = updated_lr
  return {
    "rounds_since_improvement": rounds_since_improvement,
    "previous_learning_rate": previous_lr,
    "updated_learning_rate": updated_lr,
    "changed": updated_lr < previous_lr,
  }


def run_collection_phase(config: ContinuousTrainingConfig, state: dict, teacher_policy) -> dict | None:
  episodes_to_collect = resolve_collection_increment(config, state)
  if episodes_to_collect <= 0:
    return None

  state["active_phase"] = "collect_teacher_data"
  state["status"] = "running"
  state["current_round"] = int(state.get("round_index", 0)) + 1
  save_state(config, state)

  summary = collect_trajectories(
    policy=teacher_policy,
    output_dir=config.dataset_dir,
    target_episodes=episodes_to_collect,
    seed=config.seed,
    shard_size=config.teacher_shard_size,
    append=True,
  )
  sync_dataset_state(config, state)
  state["last_collect_round"] = int(state["current_round"])
  state["last_collection_trigger_no_improvement_count"] = int(state.get("rounds_since_improvement", 0)) or None
  save_state(config, state)
  append_event(
    config,
    "teacher_collection_completed",
    {
      "episodes_collected_this_run": episodes_to_collect,
      "total_teacher_episodes": state["total_teacher_episodes"],
      "total_teacher_samples": state["total_teacher_samples"],
      "summary": summary,
    },
  )
  return summary


def run_training_phase(config: ContinuousTrainingConfig, state: dict) -> dict:
  round_index = int(state.get("round_index", 0)) + 1
  resume_model = config.student_dir / "latest_student.h5"
  if not resume_model.exists():
    legacy_resume_model = config.student_dir / "latest_student.keras"
    resume_model = legacy_resume_model if legacy_resume_model.exists() else None

  state["active_phase"] = "train_student"
  state["status"] = "running"
  state["current_round"] = round_index
  save_state(config, state)

  summary = train_student(
    DistillationConfig(
      dataset_dir=config.dataset_dir,
      output_dir=config.student_dir,
      epochs=config.student_epochs_per_round,
      batch_size=config.student_batch_size,
      learning_rate=float(state.get("learning_rate", config.student_learning_rate)),
      validation_fraction=config.student_validation_fraction,
      shuffle_buffer=config.student_shuffle_buffer,
      seed=config.seed,
      resume_model=resume_model,
      round_index=round_index,
    )
  )
  state["round_index"] = round_index
  state["total_student_rounds"] = int(state.get("total_student_rounds", 0)) + 1
  state["total_student_epochs"] = int(state.get("total_student_epochs", 0)) + config.student_epochs_per_round
  state["latest_model_path"] = summary["latest_student_path"]
  save_state(config, state)
  append_event(config, "student_training_completed", summary)
  return summary


def run_evaluation_phase(config: ContinuousTrainingConfig, state: dict) -> dict:
  round_index = int(state["round_index"])
  latest_model = Path(state["latest_model_path"])
  output_dir = config.evaluation_dir / f"round_{round_index:05d}"

  state["active_phase"] = "evaluate_student"
  state["status"] = "running"
  save_state(config, state)

  evaluation_summary = evaluate(
    config=BoxingConfig(seed=config.seed, eval_episodes=config.evaluation_episodes),
    output_dir=output_dir,
    algorithm="dqn",
    model_path=latest_model,
    episodes=config.evaluation_episodes,
    render_mode=None,
    record_video=False,
  )

  best_eval_reward = state.get("best_eval_reward")
  average_reward = evaluation_summary["average_reward"]
  if best_eval_reward is None or float(average_reward) > float(best_eval_reward):
    shutil.copy2(latest_model, best_env_model_path(config))
    with best_env_eval_path(config).open("w", encoding="utf-8") as file:
      json.dump(evaluation_summary, file, indent=2)
    state["best_eval_reward"] = average_reward
    state["best_eval_model_path"] = str(best_env_model_path(config))
    state["best_eval_summary_path"] = str(best_env_eval_path(config))
    state["rounds_since_improvement"] = 0
    state["last_collection_trigger_no_improvement_count"] = None
    state["last_lr_decay_trigger_no_improvement_count"] = None
    improved = True
  else:
    state["rounds_since_improvement"] = int(state.get("rounds_since_improvement", 0)) + 1
    improved = False

  state["last_completed_round"] = round_index
  state["status"] = "idle"
  state["active_phase"] = "idle"
  state["current_round"] = None
  save_state(config, state)
  append_event(
    config,
    "student_evaluation_completed",
    {
      "round_index": round_index,
      "improved": improved,
      "best_eval_reward": state.get("best_eval_reward"),
      "rounds_since_improvement": state.get("rounds_since_improvement"),
      "summary": evaluation_summary,
    },
  )
  return evaluation_summary


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Continuously collect teacher data and train a Keras Boxing student until a stop file is created.")
  parser.add_argument("--teacher-model", type=Path, default=Path("artifacts/teacher_cleanrl") / DEFAULT_FILENAME)
  parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/teacher_boxing_dataset_continuous"))
  parser.add_argument("--student-dir", type=Path, default=Path("artifacts/student_imitation_continuous"))
  parser.add_argument("--evaluation-dir", type=Path, default=Path("artifacts/student_imitation_continuous_eval"))
  parser.add_argument("--control-dir", type=Path, default=Path("artifacts/continuous_training"))
  parser.add_argument("--teacher-bootstrap-episodes", type=int, default=300)
  parser.add_argument("--teacher-episodes-per-collection", type=int, default=100)
  parser.add_argument("--collect-after-no-improvement-rounds", type=int, default=2)
  parser.add_argument("--teacher-shard-size", type=int, default=2048)
  parser.add_argument("--student-epochs-per-round", type=int, default=3)
  parser.add_argument("--student-batch-size", type=int, default=128)
  parser.add_argument("--student-learning-rate", type=float, default=3e-4)
  parser.add_argument("--student-validation-fraction", type=float, default=0.1)
  parser.add_argument("--student-shuffle-buffer", type=int, default=4096)
  parser.add_argument("--lr-decay-patience-rounds", type=int, default=4)
  parser.add_argument("--learning-rate-decay", type=float, default=0.5)
  parser.add_argument("--min-learning-rate", type=float, default=1e-5)
  parser.add_argument("--evaluation-episodes", type=int, default=10)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--retry-delay-seconds", type=int, default=60)
  parser.add_argument("--idle-seconds", type=int, default=5)
  return parser.parse_args()


def build_config(args: argparse.Namespace) -> ContinuousTrainingConfig:
  return ContinuousTrainingConfig(
    teacher_model=args.teacher_model,
    dataset_dir=args.dataset_dir,
    student_dir=args.student_dir,
    evaluation_dir=args.evaluation_dir,
    control_dir=args.control_dir,
    teacher_bootstrap_episodes=args.teacher_bootstrap_episodes,
    teacher_episodes_per_collection=args.teacher_episodes_per_collection,
    collect_after_no_improvement_rounds=args.collect_after_no_improvement_rounds,
    teacher_shard_size=args.teacher_shard_size,
    student_epochs_per_round=args.student_epochs_per_round,
    student_batch_size=args.student_batch_size,
    student_learning_rate=args.student_learning_rate,
    student_validation_fraction=args.student_validation_fraction,
    student_shuffle_buffer=args.student_shuffle_buffer,
    lr_decay_patience_rounds=args.lr_decay_patience_rounds,
    learning_rate_decay=args.learning_rate_decay,
    min_learning_rate=args.min_learning_rate,
    evaluation_episodes=args.evaluation_episodes,
    seed=args.seed,
    retry_delay_seconds=args.retry_delay_seconds,
    idle_seconds=args.idle_seconds,
  )


def main() -> None:
  config = build_config(parse_args())
  ensure_directories(config)
  state = load_state(config)
  sync_dataset_state(config, state)
  save_state(config, state)
  append_event(config, "supervisor_started", {"state_path": str(state_path(config))})

  teacher_policy = None
  while True:
    if stop_requested(config):
      state["status"] = "stop_requested"
      state["active_phase"] = "idle"
      state["current_round"] = None
      save_state(config, state)
      append_event(config, "supervisor_stopped", {"reason": "stop_file_detected", "stop_file": str(stop_file_path(config))})
      return

    try:
      if teacher_policy is None:
        state["active_phase"] = "load_teacher"
        state["status"] = "running"
        save_state(config, state)
        teacher_policy = load_teacher(config.teacher_model, seed=config.seed)
        append_event(config, "teacher_loaded", {"teacher_model": str(config.teacher_model)})

      run_collection_phase(config, state, teacher_policy)

      decay_summary = maybe_decay_learning_rate(config, state)
      if decay_summary is not None:
        save_state(config, state)
        append_event(config, "learning_rate_updated", decay_summary)

      run_training_phase(config, state)
      run_evaluation_phase(config, state)

      time.sleep(config.idle_seconds)
    except KeyboardInterrupt:
      state["status"] = "stopped_by_keyboard_interrupt"
      state["active_phase"] = "idle"
      state["current_round"] = None
      save_state(config, state)
      append_event(config, "supervisor_stopped", {"reason": "keyboard_interrupt"})
      return
    except Exception as exc:
      state["status"] = "error"
      state["failure_count"] = int(state.get("failure_count", 0)) + 1
      state["last_error"] = {
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "timestamp": now_iso(),
      }
      save_state(config, state)
      append_event(
        config,
        "supervisor_error",
        {
          "message": str(exc),
          "traceback": state["last_error"]["traceback"],
          "retry_delay_seconds": config.retry_delay_seconds,
        },
      )
      time.sleep(config.retry_delay_seconds)


if __name__ == "__main__":
  main()
