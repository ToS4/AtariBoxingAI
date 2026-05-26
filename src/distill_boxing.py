from __future__ import annotations

import os

BOXING_TF_DEVICE = os.environ.get("BOXING_TF_DEVICE", "cpu").strip().lower()
if BOXING_TF_DEVICE == "cpu":
  os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

if BOXING_TF_DEVICE == "gpu":
  for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)
import keras


@dataclass(slots=True)
class DistillationConfig:
  dataset_dir: Path
  output_dir: Path
  epochs: int = 5
  batch_size: int = 64
  learning_rate: float = 3e-4
  validation_fraction: float = 0.1
  shuffle_buffer: int = 4096
  seed: int = 42
  resume_model: Path | None = None
  round_index: int | None = None


def residual_block(inputs, channels: int):
  x = keras.layers.ReLU()(inputs)
  x = keras.layers.Conv2D(channels, kernel_size=3, padding="same")(x)
  x = keras.layers.ReLU()(x)
  x = keras.layers.Conv2D(channels, kernel_size=3, padding="same")(x)
  return keras.layers.Add()([x, inputs])


def conv_sequence(inputs, channels: int):
  x = keras.layers.Conv2D(channels, kernel_size=3, padding="same")(inputs)
  x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
  x = residual_block(x, channels)
  x = residual_block(x, channels)
  return x


def compile_student_model(model: keras.Model, learning_rate: float) -> keras.Model:
  model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
  )
  return model


def build_student_model(observation_shape: tuple[int, ...], action_size: int, learning_rate: float) -> keras.Model:
  inputs = keras.layers.Input(shape=observation_shape, dtype="uint8")
  x = keras.layers.Permute((2, 3, 1))(inputs)
  x = keras.layers.Rescaling(1.0 / 255.0)(x)
  for channels in (16, 32, 32):
    x = conv_sequence(x, channels)
  x = keras.layers.ReLU()(x)
  x = keras.layers.Flatten()(x)
  x = keras.layers.Dense(256, activation="relu")(x)
  logits = keras.layers.Dense(action_size, name="policy_logits")(x)
  model = keras.Model(inputs=inputs, outputs=logits)
  return compile_student_model(model, learning_rate)


def set_optimizer_learning_rate(model: keras.Model, learning_rate: float) -> None:
  optimizer = getattr(model, "optimizer", None)
  if optimizer is None:
    compile_student_model(model, learning_rate)
    return

  current_lr = optimizer.learning_rate
  if hasattr(current_lr, "assign"):
    current_lr.assign(learning_rate)
    return

  keras.backend.set_value(current_lr, learning_rate)


def load_or_build_student_model(
  observation_shape: tuple[int, ...],
  action_size: int,
  learning_rate: float,
  resume_model: Path | None,
) -> keras.Model:
  if resume_model is not None and resume_model.exists():
    model = keras.models.load_model(resume_model, compile=False)
    return compile_student_model(model, learning_rate)

  return build_student_model(observation_shape, action_size, learning_rate)
def shard_paths(dataset_dir: Path) -> list[Path]:
  return sorted(dataset_dir.glob("teacher_shard_*.npz"))


def inspect_dataset(dataset_dir: Path) -> dict:
  shard_files = shard_paths(dataset_dir)
  if not shard_files:
    raise FileNotFoundError(f"No trajectory shards found in {dataset_dir}")

  manifest_path = dataset_dir / "manifest.json"
  manifest = None
  if manifest_path.exists():
    with manifest_path.open("r", encoding="utf-8") as file:
      manifest = json.load(file)

  total_samples = 0
  total_episodes = set()
  action_histogram: dict[int, int] = {}
  observation_shape = None
  action_size = int(manifest["action_size"]) if isinstance(manifest, dict) and "action_size" in manifest else None

  for shard_file in shard_files:
    with np.load(shard_file) as shard:
      observations = shard["observations"]
      actions = shard["actions"]
      total_samples += int(actions.shape[0])
      total_episodes.update(int(item) for item in shard["episode_ids"])
      observation_shape = tuple(observations.shape[1:])
      if "logits" in shard.files:
        action_size = max(action_size or 0, int(shard["logits"].shape[1]))
      else:
        action_size = max(action_size or 0, int(np.max(actions)) + 1)
      unique, counts = np.unique(actions, return_counts=True)
      for action, count in zip(unique.tolist(), counts.tolist()):
        action_histogram[action] = action_histogram.get(action, 0) + count

  return {
    "dataset_dir": str(dataset_dir),
    "num_shards": len(shard_files),
    "num_samples": total_samples,
    "num_episodes": len(total_episodes),
    "observation_shape": observation_shape,
    "action_size": action_size,
    "action_histogram": action_histogram,
  }


def split_shards(all_shards: list[Path], validation_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
  shuffled = list(all_shards)
  random.Random(seed).shuffle(shuffled)
  if validation_fraction <= 0 or len(shuffled) < 2:
    return shuffled, []
  validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
  validation_shards = shuffled[:validation_count]
  training_shards = shuffled[validation_count:]
  if not training_shards:
    return shuffled, []
  return training_shards, validation_shards


def sample_generator(shards: list[Path]):
  for shard_file in shards:
    with np.load(shard_file) as shard:
      observations = shard["observations"]
      actions = shard["actions"]
      for index in range(actions.shape[0]):
        yield observations[index], actions[index]


def count_samples(shards: list[Path]) -> int:
  total = 0
  for shard_file in shards:
    with np.load(shard_file) as shard:
      total += int(shard["actions"].shape[0])
  return total


def build_dataset(shards: list[Path], observation_shape: tuple[int, ...], batch_size: int, shuffle_buffer: int, training: bool):
  dataset = tf.data.Dataset.from_generator(
    lambda: sample_generator(shards),
    output_signature=(
      tf.TensorSpec(shape=observation_shape, dtype=tf.uint8),
      tf.TensorSpec(shape=(), dtype=tf.int32),
    ),
  )
  if training:
    dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
  dataset = dataset.repeat()
  dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
  return dataset


def load_round_count(output_dir: Path) -> int:
  history_path = output_dir / "round_history.jsonl"
  if not history_path.exists():
    return 0

  with history_path.open("r", encoding="utf-8") as file:
    return sum(1 for line in file if line.strip())


def resolve_round_index(output_dir: Path, requested_round_index: int | None) -> int:
  if requested_round_index is not None:
    return requested_round_index
  return load_round_count(output_dir) + 1


def save_history(output_dir: Path, history: keras.callbacks.History, config: DistillationConfig, dataset_summary: dict, round_index: int) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  payload = {
    "config": {
      **asdict(config),
      "dataset_dir": str(config.dataset_dir),
      "output_dir": str(config.output_dir),
      "resume_model": str(config.resume_model) if config.resume_model is not None else None,
    },
    "dataset_summary": dataset_summary,
    "round_index": round_index,
    "history": history.history,
  }
  history_path = output_dir / "history.json"
  with history_path.open("w", encoding="utf-8") as file:
    json.dump(payload, file, indent=2)

  round_history_path = output_dir / f"history_round_{round_index:05d}.json"
  with round_history_path.open("w", encoding="utf-8") as file:
    json.dump(payload, file, indent=2)

  fig, axes = plt.subplots(1, 2, figsize=(12, 4))
  axes[0].plot(history.history["loss"], label="train")
  if "val_loss" in history.history:
    axes[0].plot(history.history["val_loss"], label="val")
  axes[0].set_title("Loss")
  axes[0].set_xlabel("Epoch")
  axes[0].legend()

  axes[1].plot(history.history["accuracy"], label="train")
  if "val_accuracy" in history.history:
    axes[1].plot(history.history["val_accuracy"], label="val")
  axes[1].set_title("Accuracy")
  axes[1].set_xlabel("Epoch")
  axes[1].legend()

  fig.tight_layout()
  fig.savefig(output_dir / "training_curves.png", dpi=150)
  fig.savefig(output_dir / f"training_curves_round_{round_index:05d}.png", dpi=150)
  plt.close(fig)


def append_round_history(output_dir: Path, summary: dict) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  with (output_dir / "round_history.jsonl").open("a", encoding="utf-8") as file:
    file.write(json.dumps(summary) + "\n")


def train_student(config: DistillationConfig) -> dict:
  tf.random.set_seed(config.seed)
  np.random.seed(config.seed)
  random.seed(config.seed)

  config.output_dir.mkdir(parents=True, exist_ok=True)
  round_index = resolve_round_index(config.output_dir, config.round_index)

  dataset_summary = inspect_dataset(config.dataset_dir)
  all_shards = shard_paths(config.dataset_dir)
  training_shards, validation_shards = split_shards(all_shards, config.validation_fraction, config.seed)
  observation_shape = tuple(dataset_summary["observation_shape"])
  action_size = int(dataset_summary["action_size"])

  training_dataset = build_dataset(
    training_shards,
    observation_shape=observation_shape,
    batch_size=config.batch_size,
    shuffle_buffer=config.shuffle_buffer,
    training=True,
  )
  validation_dataset = None
  train_samples = count_samples(training_shards)
  validation_samples = count_samples(validation_shards)
  train_steps = max(1, math.ceil(train_samples / config.batch_size))
  validation_steps = max(1, math.ceil(validation_samples / config.batch_size)) if validation_shards else None
  if validation_shards:
    validation_dataset = build_dataset(
      validation_shards,
      observation_shape=observation_shape,
      batch_size=config.batch_size,
      shuffle_buffer=config.shuffle_buffer,
      training=False,
    )

  latest_model_path = config.output_dir / "latest_student.h5"
  legacy_latest_model_path = config.output_dir / "latest_student.keras"
  resume_model = config.resume_model
  if resume_model is None:
    if latest_model_path.exists():
      resume_model = latest_model_path
    elif legacy_latest_model_path.exists():
      resume_model = legacy_latest_model_path

  model = load_or_build_student_model(
    observation_shape=observation_shape,
    action_size=action_size,
    learning_rate=config.learning_rate,
    resume_model=resume_model,
  )

  best_model_path = config.output_dir / "best_student.h5"
  final_model_path = config.output_dir / "final_student.h5"
  backup_dir = config.output_dir / "backup"
  monitor = "val_accuracy" if validation_dataset is not None else "accuracy"
  callbacks: list[keras.callbacks.Callback] = [
    keras.callbacks.BackupAndRestore(backup_dir=str(backup_dir)),
    keras.callbacks.ModelCheckpoint(
      filepath=str(latest_model_path),
      save_best_only=False,
    ),
    keras.callbacks.ModelCheckpoint(
      filepath=str(best_model_path),
      monitor=monitor,
      mode="max",
      save_best_only=True,
    ),
  ]

  history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    steps_per_epoch=train_steps,
    validation_steps=validation_steps,
    epochs=config.epochs,
    callbacks=callbacks,
    verbose=2,
  )

  model.save(final_model_path)
  model.save(latest_model_path)
  model.save(config.output_dir / "final_student.keras")
  model.save(config.output_dir / "latest_student.keras")
  if best_model_path.exists():
    try:
      best_model = keras.models.load_model(best_model_path)
      best_model.save(config.output_dir / "best_student.keras")
    except Exception:
      pass
  save_history(config.output_dir, history, config, dataset_summary, round_index)

  summary = {
    "round_index": round_index,
    "dataset_summary": dataset_summary,
    "train_samples": train_samples,
    "validation_samples": validation_samples,
    "resume_model": str(resume_model) if resume_model is not None else None,
    "latest_student_path": str(latest_model_path),
    "best_student_path": str(best_model_path),
    "final_student_path": str(final_model_path),
    "history_path": str(config.output_dir / "history.json"),
    "round_history_path": str(config.output_dir / f"history_round_{round_index:05d}.json"),
    "final_metrics": {key: float(values[-1]) for key, values in history.history.items()},
  }
  with (config.output_dir / "training_summary.json").open("w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)
  append_round_history(config.output_dir, summary)
  return summary


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Train a Keras student on teacher Boxing trajectories.")
  subparsers = parser.add_subparsers(dest="command", required=True)

  inspect_parser = subparsers.add_parser("inspect", help="Inspect a saved teacher trajectory dataset.")
  inspect_parser.add_argument("--dataset-dir", type=Path, required=True)

  train_parser = subparsers.add_parser("train", help="Train a Keras student from teacher trajectories.")
  train_parser.add_argument("--dataset-dir", type=Path, required=True)
  train_parser.add_argument("--output-dir", type=Path, required=True)
  train_parser.add_argument("--epochs", type=int, default=5)
  train_parser.add_argument("--batch-size", type=int, default=64)
  train_parser.add_argument("--learning-rate", type=float, default=3e-4)
  train_parser.add_argument("--validation-fraction", type=float, default=0.1)
  train_parser.add_argument("--shuffle-buffer", type=int, default=4096)
  train_parser.add_argument("--seed", type=int, default=42)
  train_parser.add_argument("--resume-model", type=Path, default=None)
  train_parser.add_argument("--round-index", type=int, default=None)

  return parser.parse_args()


def main() -> None:
  args = parse_args()

  if args.command == "inspect":
    summary = inspect_dataset(args.dataset_dir)
    print(json.dumps(summary, indent=2))
    return

  if args.command == "train":
    config = DistillationConfig(
      dataset_dir=args.dataset_dir,
      output_dir=args.output_dir,
      epochs=args.epochs,
      batch_size=args.batch_size,
      learning_rate=args.learning_rate,
      validation_fraction=args.validation_fraction,
      shuffle_buffer=args.shuffle_buffer,
      seed=args.seed,
      resume_model=args.resume_model,
      round_index=args.round_index,
    )
    summary = train_student(config)
    print(json.dumps(summary, indent=2))
    return

  raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
  main()
