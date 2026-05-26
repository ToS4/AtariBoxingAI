from __future__ import annotations

import os

BOXING_TF_DEVICE = os.environ.get("BOXING_TF_DEVICE", "cpu").strip().lower()
if BOXING_TF_DEVICE == "cpu":
  os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
from datetime import datetime, timezone
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
try:
  from tqdm import tqdm
except ImportError:  # pragma: no cover - handled at runtime
  def tqdm(iterable, **kwargs):
    del kwargs
    return iterable

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ML_IMPORT_ERROR: Exception | None = None
try:
  import keras
  import tensorflow as tf

  if BOXING_TF_DEVICE == "gpu":
    for gpu in tf.config.list_physical_devices("GPU"):
      tf.config.experimental.set_memory_growth(gpu, True)
except ImportError as exc:  # pragma: no cover - handled at runtime
  keras = None
  tf = None
  _ML_IMPORT_ERROR = exc

_ATARI_IMPORT_ERROR: Exception | None = None
try:
  import ale_py
  import gymnasium as gym
  from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo
except ImportError as exc:  # pragma: no cover - handled at runtime
  ale_py = None
  gym = None
  AtariPreprocessing = None
  FrameStackObservation = None
  RecordVideo = None
  _ATARI_IMPORT_ERROR = exc


@dataclass(slots=True)
class BoxingConfig:
  env_id: str = "ALE/Boxing-v5"
  episodes: int = 300
  eval_episodes: int = 5
  selection_eval_episodes: int = 20
  selection_eval_frequency: int = 25
  learning_rate: float = 1e-4
  discount_factor: float = 0.99
  batch_size: int = 32
  memory_size: int = 50_000
  epsilon_start: float = 1.0
  epsilon_end: float = 0.1
  target_update_freq: int = 1_000
  checkpoint_freq: int = 25
  warmup_steps: int = 2_000
  train_frequency: int = 4
  screen_size: int = 84
  stack_size: int = 4
  frame_skip: int = 4
  noop_max: int = 30
  full_action_space: bool = True
  repeat_action_probability: float = 0.25
  seed: int = 42


def _require_ml_stack() -> None:
  if keras is None or tf is None:
    raise RuntimeError(
      "TensorFlow / Keras are required. Install the packages from requirements.txt before running boxing.py."
    ) from _ML_IMPORT_ERROR


def _require_atari_stack() -> None:
  if gym is None or ale_py is None:
    raise RuntimeError(
      "Gymnasium Atari support is required. Install the packages from requirements.txt before running boxing.py."
    ) from _ATARI_IMPORT_ERROR

  try:
    import cv2  # noqa: F401
  except ImportError as exc:
    raise RuntimeError(
      "opencv-python is required for Atari preprocessing. Install the packages from requirements.txt first."
    ) from exc


def _register_ale() -> None:
  _require_atari_stack()
  gym.register_envs(ale_py)


def compute_ema(values: list[float], alpha: float = 0.05) -> list[float]:
  ema_values: list[float] = []
  current: float | None = None
  for value in values:
    current = value if current is None else ((1.0 - alpha) * current + alpha * value)
    ema_values.append(float(current))
  return ema_values


def build_model(observation_shape: tuple[int, ...], action_size: int, learning_rate: float):
  _require_ml_stack()

  if len(observation_shape) != 3:
    raise ValueError(f"Expected stacked grayscale observations with shape (stack, height, width), got {observation_shape}.")

  model = keras.Sequential([
    keras.layers.Input(shape=observation_shape, dtype="uint8"),
    # FrameStackObservation produces (stack, H, W); Conv2D expects channels last on CPU.
    keras.layers.Permute((2, 3, 1)),
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(32, kernel_size=8, strides=4, activation="relu"),
    keras.layers.Conv2D(64, kernel_size=4, strides=2, activation="relu"),
    keras.layers.Conv2D(64, kernel_size=3, strides=1, activation="relu"),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(action_size),
  ])
  model.compile(
    loss=keras.losses.Huber(),
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
  )
  return model


def make_boxing_env(
  config: BoxingConfig,
  render_mode: str | None = None,
  video_dir: Path | None = None,
):
  _register_ale()

  env = gym.make(
    config.env_id,
    render_mode=render_mode,
    frameskip=1,
    full_action_space=config.full_action_space,
    repeat_action_probability=config.repeat_action_probability,
  )
  env = AtariPreprocessing(
    env,
    noop_max=config.noop_max,
    frame_skip=config.frame_skip,
    screen_size=config.screen_size,
    terminal_on_life_loss=False,
    grayscale_obs=True,
    grayscale_newaxis=False,
    scale_obs=False,
  )
  env = FrameStackObservation(env, stack_size=config.stack_size)

  if video_dir is not None:
    video_dir.mkdir(parents=True, exist_ok=True)
    env = RecordVideo(
      env,
      video_folder=str(video_dir),
      episode_trigger=lambda _: True,
      name_prefix="boxing-eval",
      disable_logger=True,
    )

  return env


def load_json_file(path: Path, default):
  if not path.exists():
    return default

  with path.open("r", encoding="utf-8") as file:
    return json.load(file)


def append_experiment_log(output_dir: Path, event: str, payload: dict) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  record = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "event": event,
    **payload,
  }
  with (output_dir / "experiment_log.jsonl").open("a", encoding="utf-8") as file:
    file.write(json.dumps(record) + "\n")


def summarize_rewards(rewards: list[float]) -> dict:
  return {
    "episodes": rewards,
    "average_reward": float(np.mean(rewards)) if rewards else 0.0,
    "std_reward": float(np.std(rewards)) if rewards else 0.0,
  }


def rollout_policy(agent, env, episodes: int, seed_base: int) -> list[float]:
  rewards: list[float] = []
  for episode in range(episodes):
    state, _ = env.reset(seed=seed_base + episode)
    terminated = False
    truncated = False
    episode_reward = 0.0

    while not (terminated or truncated):
      action = agent.act(state, train=False)
      state, reward, terminated, truncated, _ = env.step(action)
      episode_reward += float(reward)

    rewards.append(float(episode_reward))

  return rewards


def evaluate_agent_policy(
  agent,
  config: BoxingConfig,
  episodes: int,
  render_mode: str | None = None,
  video_dir: Path | None = None,
  seed_base: int | None = None,
) -> dict:
  env = make_boxing_env(config=config, render_mode=render_mode, video_dir=video_dir)
  try:
    rewards = rollout_policy(agent, env, episodes=episodes, seed_base=seed_base or (config.seed + 10_000))
  finally:
    env.close()

  return summarize_rewards(rewards)


def load_evaluation_history(output_dir: Path) -> list[dict]:
  return list(load_json_file(output_dir / "evaluation_history.json", []))


def load_best_evaluation(output_dir: Path) -> dict | None:
  best_evaluation = load_json_file(output_dir / "best_evaluation.json", None)
  return best_evaluation if isinstance(best_evaluation, dict) else None


def save_evaluation_artifacts(output_dir: Path, evaluation_history: list[dict], best_evaluation: dict | None) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)

  with (output_dir / "evaluation_history.json").open("w", encoding="utf-8") as file:
    json.dump(evaluation_history, file, indent=2)

  if best_evaluation is not None:
    with (output_dir / "best_evaluation.json").open("w", encoding="utf-8") as file:
      json.dump(best_evaluation, file, indent=2)

  if not evaluation_history:
    return

  eval_episodes = [entry["episode_index"] for entry in evaluation_history]
  avg_rewards = [entry["average_reward"] for entry in evaluation_history]
  std_rewards = [entry["std_reward"] for entry in evaluation_history]

  fig, ax = plt.subplots(figsize=(6, 4))
  ax.plot(eval_episodes, avg_rewards, marker="o", label="Average reward")
  lower = [avg - std for avg, std in zip(avg_rewards, std_rewards)]
  upper = [avg + std for avg, std in zip(avg_rewards, std_rewards)]
  ax.fill_between(eval_episodes, lower, upper, alpha=0.2, label="+/- 1 std")
  ax.set_title("Selection Evaluations")
  ax.set_xlabel("Training episode")
  ax.set_ylabel("Reward")
  ax.legend()
  fig.tight_layout()
  fig.savefig(output_dir / "selection_curve.png", dpi=150)
  plt.close(fig)


def _agent_modules():
  from agent.dqn_agent import DQN_Agent_Off_Policy
  from agent.ddqn_agent import DDQN_Agent_Off_Policy
  from exploration.exploration import EGreedyExp
  from replay_memory.replay_memory import ExpReplay, PrioReplay

  return DQN_Agent_Off_Policy, DDQN_Agent_Off_Policy, EGreedyExp, ExpReplay, PrioReplay


def create_agent(
  config: BoxingConfig,
  observation_shape: tuple[int, ...],
  action_size: int,
  algorithm: str,
  prioritized_replay: bool,
):
  DQN_Agent_Off_Policy, DDQN_Agent_Off_Policy, EGreedyExp, ExpReplay, PrioReplay = _agent_modules()

  exploration = EGreedyExp(
    epsilon_start=config.epsilon_start,
    epsilon_end=config.epsilon_end,
    episodes_count=config.episodes,
    action_size=action_size,
  )
  memory = PrioReplay(size=config.memory_size) if prioritized_replay else ExpReplay(size=config.memory_size)
  model_fn = lambda: build_model(observation_shape, action_size, config.learning_rate)

  if algorithm == "ddqn":
    agent = DDQN_Agent_Off_Policy(
      model_fn=model_fn,
      action_size=action_size,
      exploration=exploration,
      batch_size=config.batch_size,
      learning_rate=config.learning_rate,
      memory=memory,
      discount_factor=config.discount_factor,
      target_update_freq=config.target_update_freq,
    )
  elif algorithm == "dqn":
    agent = DQN_Agent_Off_Policy(
      model_fn=model_fn,
      action_size=action_size,
      exploration=exploration,
      batch_size=config.batch_size,
      learning_rate=config.learning_rate,
      memory=memory,
      discount_factor=config.discount_factor,
    )
  else:
    raise ValueError(f"Unsupported algorithm: {algorithm}")

  return agent, memory


def load_training_history(output_dir: Path) -> dict:
  return load_json_file(
    output_dir / "training_history.json",
    {
      "episode_rewards": [],
      "ema_rewards": [],
      "epsilon_history": [],
      "total_steps": 0,
    },
  )


def save_training_artifacts(
  output_dir: Path,
  config: BoxingConfig,
  algorithm: str,
  prioritized_replay: bool,
  episode_rewards: list[float],
  epsilon_history: list[float],
  total_steps: int,
) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)

  ema_rewards = compute_ema(episode_rewards)
  history = {
    "algorithm": algorithm,
    "prioritized_replay": prioritized_replay,
    "config": asdict(config),
    "episode_rewards": episode_rewards,
    "ema_rewards": ema_rewards,
    "epsilon_history": epsilon_history,
    "total_steps": total_steps,
  }

  with (output_dir / "training_history.json").open("w", encoding="utf-8") as file:
    json.dump(history, file, indent=2)

  fig, axes = plt.subplots(1, 2, figsize=(12, 4))

  axes[0].plot(episode_rewards, label="Episode reward", alpha=0.5)
  axes[0].plot(ema_rewards, label="EMA reward", linewidth=2)
  axes[0].set_title("Training Rewards")
  axes[0].set_xlabel("Episode")
  axes[0].set_ylabel("Reward")
  axes[0].legend()

  axes[1].plot(epsilon_history)
  axes[1].set_title("Exploration")
  axes[1].set_xlabel("Episode")
  axes[1].set_ylabel("Epsilon")

  fig.tight_layout()
  fig.savefig(output_dir / "training_curve.png", dpi=150)
  plt.close(fig)

  with (output_dir / "training_config.json").open("w", encoding="utf-8") as file:
    json.dump(
      {
        "algorithm": algorithm,
        "prioritized_replay": prioritized_replay,
        "config": asdict(config),
      },
      file,
      indent=2,
    )


def save_checkpoint(agent, checkpoint_dir: Path, episode: int, total_steps: int, algorithm: str, config: BoxingConfig) -> None:
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  agent.save_checkpoint(str(checkpoint_dir))
  with (checkpoint_dir / "meta.json").open("w", encoding="utf-8") as file:
    json.dump(
      {
        "episode": episode,
        "total_steps": total_steps,
        "algorithm": algorithm,
        "config": asdict(config),
      },
      file,
      indent=2,
    )


def load_checkpoint_metadata(checkpoint_dir: Path) -> dict:
  with (checkpoint_dir / "meta.json").open("r", encoding="utf-8") as file:
    return json.load(file)


def default_training_model_path(output_dir: Path = Path("artifacts/boxing_ddqn")) -> Path:
  best_model_path = output_dir / "best_model.keras"
  if best_model_path.exists():
    return best_model_path
  return output_dir / "final_model.keras"


def train(
  config: BoxingConfig,
  output_dir: Path,
  algorithm: str = "ddqn",
  prioritized_replay: bool = True,
  resume_from: Path | None = None,
):
  _require_ml_stack()
  _require_atari_stack()

  random.seed(config.seed)
  np.random.seed(config.seed)
  tf.random.set_seed(config.seed)

  env = make_boxing_env(config=config, render_mode=None)
  observation_shape = tuple(env.observation_space.shape)
  action_size = int(env.action_space.n)
  agent, memory = create_agent(config, observation_shape, action_size, algorithm, prioritized_replay)

  history = load_training_history(output_dir)
  episode_rewards: list[float] = list(history.get("episode_rewards", []))
  epsilon_history: list[float] = list(history.get("epsilon_history", []))
  total_steps = int(history.get("total_steps", 0))
  start_episode = len(episode_rewards)
  evaluation_history = load_evaluation_history(output_dir)
  best_evaluation = load_best_evaluation(output_dir)
  best_average_reward = (
    float(best_evaluation["average_reward"])
    if best_evaluation is not None and best_evaluation.get("average_reward") is not None
    else -np.inf
  )
  eval_seed_base = config.seed + 100_000

  append_experiment_log(
    output_dir,
    "training_started",
    {
      "algorithm": algorithm,
      "prioritized_replay": prioritized_replay,
      "resume_from": str(resume_from) if resume_from is not None else None,
      "config": asdict(config),
      "existing_training_episodes": len(episode_rewards),
      "existing_selection_evaluations": len(evaluation_history),
    },
  )

  if resume_from is not None:
    metadata = load_checkpoint_metadata(resume_from)
    agent.load_checkpoint(str(resume_from))
    start_episode = int(metadata["episode"]) + 1
    total_steps = int(metadata.get("total_steps", total_steps))
    print("Checkpoint loaded. Replay memory is not restored; the buffer will refill during resumed training.")
    append_experiment_log(
      output_dir,
      "training_resumed",
      {
        "resume_from": str(resume_from),
        "start_episode": start_episode,
        "total_steps": total_steps,
      },
    )

  pbar = tqdm(range(start_episode, config.episodes), desc=f"Training {algorithm.upper()}")
  for episode in pbar:
    state, _ = env.reset(seed=config.seed + episode)
    episode_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
      action = agent.act(state, train=True)
      next_state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated

      agent.remember(state=state, action=action, next_state=next_state, reward=reward, done=done)

      total_steps += 1
      if len(memory) >= config.warmup_steps and total_steps % config.train_frequency == 0:
        agent.train()

      state = next_state
      episode_reward += float(reward)

    agent.after_episode()
    episode_rewards.append(float(episode_reward))

    epsilon = getattr(agent._exploration, "epsilon", 0.0)
    epsilon_history.append(float(epsilon))

    ema_reward = compute_ema(episode_rewards)[-1]
    pbar.set_postfix({
      "reward": f"{episode_reward:.1f}",
      "ema": f"{ema_reward:.1f}",
      "eps": f"{epsilon:.3f}",
      "mem": len(memory),
    })

    if (episode + 1) % config.checkpoint_freq == 0:
      checkpoint_dir = output_dir / "checkpoints" / f"ep_{episode + 1:05d}"
      save_checkpoint(agent, checkpoint_dir, episode, total_steps, algorithm, config)
      save_training_artifacts(output_dir, config, algorithm, prioritized_replay, episode_rewards, epsilon_history, total_steps)
      append_experiment_log(
        output_dir,
        "checkpoint_saved",
        {
          "episode": episode,
          "episode_index": episode + 1,
          "total_steps": total_steps,
          "checkpoint_dir": str(checkpoint_dir),
        },
      )

    should_run_selection_eval = (
      config.selection_eval_frequency > 0
      and ((episode + 1) % config.selection_eval_frequency == 0 or (episode + 1) == config.episodes)
    )
    if should_run_selection_eval:
      selection_summary = evaluate_agent_policy(
        agent=agent,
        config=config,
        episodes=config.selection_eval_episodes,
        seed_base=eval_seed_base,
      )
      selection_entry = {
        "episode": episode,
        "episode_index": episode + 1,
        "total_steps": total_steps,
        "num_eval_episodes": config.selection_eval_episodes,
        **selection_summary,
      }
      evaluation_history.append(selection_entry)
      append_experiment_log(output_dir, "selection_evaluation", selection_entry)

      if selection_entry["average_reward"] > best_average_reward:
        best_average_reward = selection_entry["average_reward"]
        best_model_path = output_dir / "best_model.keras"
        best_checkpoint_dir = output_dir / "best_checkpoint"
        agent.save_model(str(best_model_path))
        save_checkpoint(agent, best_checkpoint_dir, episode, total_steps, algorithm, config)
        best_evaluation = {
          **selection_entry,
          "model_path": str(best_model_path),
          "checkpoint_path": str(best_checkpoint_dir),
        }
        append_experiment_log(output_dir, "best_model_updated", best_evaluation)

      save_evaluation_artifacts(output_dir, evaluation_history, best_evaluation)

      pbar.set_postfix({
        "reward": f"{episode_reward:.1f}",
        "ema": f"{ema_reward:.1f}",
        "eps": f"{epsilon:.3f}",
        "mem": len(memory),
        "best": f"{best_average_reward:.1f}" if best_average_reward > -np.inf else "-",
      })

  env.close()

  output_dir.mkdir(parents=True, exist_ok=True)
  final_model_path = output_dir / "final_model.keras"
  agent.save_model(str(final_model_path))
  final_checkpoint = output_dir / "checkpoints" / f"ep_{config.episodes:05d}"
  save_checkpoint(agent, final_checkpoint, config.episodes - 1, total_steps, algorithm, config)
  save_training_artifacts(output_dir, config, algorithm, prioritized_replay, episode_rewards, epsilon_history, total_steps)
  if best_evaluation is None:
    best_model_path = output_dir / "best_model.keras"
    best_checkpoint_dir = output_dir / "best_checkpoint"
    agent.save_model(str(best_model_path))
    save_checkpoint(agent, best_checkpoint_dir, config.episodes - 1, total_steps, algorithm, config)
    best_evaluation = {
      "episode": config.episodes - 1,
      "episode_index": config.episodes,
      "total_steps": total_steps,
      "num_eval_episodes": 0,
      "episodes": [],
      "average_reward": None,
      "std_reward": None,
      "selection_disabled": config.selection_eval_frequency <= 0,
      "model_path": str(best_model_path),
      "checkpoint_path": str(best_checkpoint_dir),
    }
  save_evaluation_artifacts(output_dir, evaluation_history, best_evaluation)
  append_experiment_log(
    output_dir,
    "training_finished",
    {
      "final_model_path": str(final_model_path),
      "final_checkpoint": str(final_checkpoint),
      "best_model_path": str(output_dir / "best_model.keras"),
      "best_average_reward": best_evaluation.get("average_reward"),
      "selection_evaluations": len(evaluation_history),
    },
  )
  return agent


def load_agent_for_inference(
  config: BoxingConfig,
  observation_shape: tuple[int, ...],
  action_size: int,
  algorithm: str,
  model_path: Path | None = None,
  checkpoint_path: Path | None = None,
):
  agent, _ = create_agent(config, observation_shape, action_size, algorithm, prioritized_replay=True)

  if checkpoint_path is not None:
    agent.load_checkpoint(str(checkpoint_path))
    return agent

  if model_path is None:
    raise ValueError("Either model_path or checkpoint_path must be provided.")

  agent.load_model(str(model_path))
  return agent


def evaluate(
  config: BoxingConfig,
  output_dir: Path,
  algorithm: str = "ddqn",
  model_path: Path | None = None,
  checkpoint_path: Path | None = None,
  episodes: int | None = None,
  render_mode: str | None = None,
  record_video: bool = False,
):
  _require_ml_stack()
  _require_atari_stack()

  eval_episodes = episodes or config.eval_episodes
  effective_render_mode = "rgb_array" if record_video else render_mode
  video_dir = output_dir / "evaluation_videos" if record_video else None

  env = make_boxing_env(config=config, render_mode=effective_render_mode, video_dir=video_dir)
  observation_shape = tuple(env.observation_space.shape)
  action_size = int(env.action_space.n)
  agent = load_agent_for_inference(
    config=config,
    observation_shape=observation_shape,
    action_size=action_size,
    algorithm=algorithm,
    model_path=model_path,
    checkpoint_path=checkpoint_path,
  )

  try:
    rewards = rollout_policy(agent, env, episodes=eval_episodes, seed_base=config.seed + 10_000)
  finally:
    env.close()
  summary = {
    "algorithm": algorithm,
    **summarize_rewards(rewards),
    "model_path": str(model_path) if model_path is not None else None,
    "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
  }

  output_dir.mkdir(parents=True, exist_ok=True)
  with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

  print(json.dumps(summary, indent=2))
  return summary


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Train and evaluate a DQN / DDQN Boxing agent.")
  subparsers = parser.add_subparsers(dest="command", required=True)

  train_parser = subparsers.add_parser("train", help="Train a Boxing agent.")
  train_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/boxing_ddqn"))
  train_parser.add_argument("--episodes", type=int, default=300)
  train_parser.add_argument("--algorithm", choices=("dqn", "ddqn"), default="ddqn")
  train_parser.add_argument("--uniform-replay", action="store_true", help="Use uniform replay instead of prioritized replay.")
  train_parser.add_argument("--resume-from", type=Path, default=None)
  train_parser.add_argument("--seed", type=int, default=42)
  train_parser.add_argument("--checkpoint-freq", type=int, default=25)
  train_parser.add_argument("--warmup-steps", type=int, default=2_000)
  train_parser.add_argument("--selection-eval-frequency", type=int, default=25, help="Run internal model-selection evaluation every N training episodes.")
  train_parser.add_argument("--selection-eval-episodes", type=int, default=20, help="Number of built-in-opponent games per internal selection evaluation.")

  evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a saved agent against the built-in Atari opponent.")
  evaluate_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/boxing_eval"))
  evaluate_parser.add_argument("--algorithm", choices=("dqn", "ddqn"), default="ddqn")
  evaluate_parser.add_argument("--model", type=Path, default=None)
  evaluate_parser.add_argument("--checkpoint", type=Path, default=None)
  evaluate_parser.add_argument("--episodes", type=int, default=5)
  evaluate_parser.add_argument("--video", action="store_true", help="Record the evaluation episodes as videos.")

  play_parser = subparsers.add_parser("play", help="Play a saved agent in a visible Boxing window.")
  play_parser.add_argument("--algorithm", choices=("dqn", "ddqn"), default="ddqn")
  play_parser.add_argument("--model", type=Path, default=None)
  play_parser.add_argument("--checkpoint", type=Path, default=None)
  play_parser.add_argument("--episodes", type=int, default=1)

  return parser.parse_args()


def main() -> None:
  args = parse_args()
  config = BoxingConfig(
    episodes=getattr(args, "episodes", 300),
    checkpoint_freq=getattr(args, "checkpoint_freq", 25),
    warmup_steps=getattr(args, "warmup_steps", 2_000),
    selection_eval_frequency=getattr(args, "selection_eval_frequency", 25),
    selection_eval_episodes=getattr(args, "selection_eval_episodes", 20),
    seed=getattr(args, "seed", 42),
  )

  if args.command == "train":
    train(
      config=config,
      output_dir=args.output_dir,
      algorithm=args.algorithm,
      prioritized_replay=not args.uniform_replay,
      resume_from=args.resume_from,
    )
    return

  if args.command == "evaluate":
    if args.model is None and args.checkpoint is None:
      args.model = default_training_model_path()

    evaluate(
      config=config,
      output_dir=args.output_dir,
      algorithm=args.algorithm,
      model_path=args.model,
      checkpoint_path=args.checkpoint,
      episodes=args.episodes,
      render_mode=None,
      record_video=args.video,
    )
    return

  if args.command == "play":
    if args.model is None and args.checkpoint is None:
      args.model = default_training_model_path()

    evaluate(
      config=config,
      output_dir=Path("artifacts/boxing_play"),
      algorithm=args.algorithm,
      model_path=args.model,
      checkpoint_path=args.checkpoint,
      episodes=args.episodes,
      render_mode="human",
      record_video=False,
    )
    return

  raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
  main()
