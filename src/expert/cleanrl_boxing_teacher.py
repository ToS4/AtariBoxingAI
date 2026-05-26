from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from dataclasses import dataclass
from pathlib import Path

import envpool
import flax.linen as nn
from flax import serialization
from flax.linen.initializers import constant, orthogonal
from huggingface_hub import hf_hub_download
import jax
import jax.numpy as jnp
import numpy as np


DEFAULT_REPO_ID = "cleanrl/Boxing-v5-cleanba_ppo_envpool_impala_atari_wrapper-seed2"
DEFAULT_FILENAME = "cleanba_ppo_envpool_impala_atari_wrapper.cleanrl_model"
ATARI_MAX_FRAMES = int(108000 / 4)


class ResidualBlock(nn.Module):
  channels: int

  @nn.compact
  def __call__(self, x):
    inputs = x
    x = nn.relu(x)
    x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
    return x + inputs


class ConvSequence(nn.Module):
  channels: int

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
    x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
    x = ResidualBlock(self.channels)(x)
    x = ResidualBlock(self.channels)(x)
    return x


class Network(nn.Module):
  channelss: tuple[int, ...] = (16, 32, 32)

  @nn.compact
  def __call__(self, x):
    x = jnp.transpose(x, (0, 2, 3, 1))
    x = x / 255.0
    for channels in self.channelss:
      x = ConvSequence(channels)(x)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
    x = nn.relu(x)
    return x


class Actor(nn.Module):
  action_dim: int

  @nn.compact
  def __call__(self, x):
    return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


class Critic(nn.Module):
  @nn.compact
  def __call__(self, x):
    return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


@dataclass(slots=True)
class TeacherPolicy:
  model_path: Path
  network_params: object
  actor_params: object
  critic_params: object
  action_size: int

  def predict_logits(self, observation: np.ndarray) -> np.ndarray:
    hidden = Network().apply(self.network_params, observation)
    logits = Actor(action_dim=self.action_size).apply(self.actor_params, hidden)
    return np.asarray(logits, dtype=np.float32)

  def greedy_action(self, observation: np.ndarray) -> np.ndarray:
    logits = self.predict_logits(observation)
    return np.argmax(logits, axis=1).astype(np.int32)


def build_env(seed: int):
  return envpool.make(
    "Boxing-v5",
    env_type="gymnasium",
    num_envs=1,
    batch_size=1,
    episodic_life=True,
    repeat_action_probability=0,
    noop_max=30,
    full_action_space=False,
    max_episode_steps=ATARI_MAX_FRAMES,
    reward_clip=True,
    seed=seed,
  )


def download_teacher(repo_id: str, filename: str, local_dir: Path) -> Path:
  local_dir.mkdir(parents=True, exist_ok=True)
  downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(local_dir))
  return Path(downloaded_path)


def load_teacher(model_path: Path, seed: int) -> TeacherPolicy:
  env = build_env(seed=seed)
  try:
    observation, _ = env.reset()
    sample_observation = np.asarray(observation[:1])
    action_size = int(env.action_space.n)
  finally:
    env.close()

  network = Network()
  actor = Actor(action_dim=action_size)
  critic = Critic()
  network_params = network.init(jax.random.PRNGKey(0), sample_observation)
  hidden = network.apply(network_params, sample_observation)
  actor_params = actor.init(jax.random.PRNGKey(1), hidden)
  critic_params = critic.init(jax.random.PRNGKey(2), hidden)

  raw_bytes = model_path.read_bytes()
  _, params_list = serialization.from_bytes([dict(), [network_params, actor_params, critic_params]], raw_bytes)
  network_params, actor_params, critic_params = params_list
  return TeacherPolicy(
    model_path=model_path,
    network_params=network_params,
    actor_params=actor_params,
    critic_params=critic_params,
    action_size=action_size,
  )


def evaluate_teacher(policy: TeacherPolicy, episodes: int, seed: int) -> dict:
  env = build_env(seed=seed)
  returns: list[float] = []
  current_return = 0.0

  try:
    observation, _ = env.reset()
    while len(returns) < episodes:
      observation = np.asarray(observation)
      action = policy.greedy_action(observation)
      observation, reward, terminated, truncated, _ = env.step(action)
      current_return += float(reward[0])
      if bool(terminated[0]) or bool(truncated[0]):
        returns.append(current_return)
        current_return = 0.0
        observation, _ = env.reset()
  finally:
    env.close()

  return {
    "episodes": returns,
    "average_reward": float(np.mean(returns)) if returns else 0.0,
    "std_reward": float(np.std(returns)) if returns else 0.0,
    "num_episodes": len(returns),
    "model_path": str(policy.model_path),
  }


def _timestamp() -> str:
  return datetime.now(timezone.utc).isoformat()


def _manifest_path(output_dir: Path) -> Path:
  return output_dir / "manifest.json"


def _existing_shard_paths(output_dir: Path) -> list[Path]:
  return sorted(output_dir.glob("teacher_shard_*.npz"))


def _load_existing_manifest(output_dir: Path) -> dict | None:
  manifest_path = _manifest_path(output_dir)
  if not manifest_path.exists():
    return None

  with manifest_path.open("r", encoding="utf-8") as file:
    return json.load(file)


def _recover_existing_progress(output_dir: Path) -> dict:
  shard_paths = _existing_shard_paths(output_dir)
  if not shard_paths:
    return {
      "num_episodes": 0,
      "num_samples": 0,
      "episodic_returns": [],
    }

  num_samples = 0
  max_episode_id = -1
  for shard_path in shard_paths:
    with np.load(shard_path) as shard:
      num_samples += int(shard["actions"].shape[0])
      if shard["episode_ids"].size:
        max_episode_id = max(max_episode_id, int(np.max(shard["episode_ids"])))

  return {
    "num_episodes": max_episode_id + 1,
    "num_samples": num_samples,
    "episodic_returns": [],
  }


def _count_samples(output_dir: Path) -> int:
  sample_count = 0
  for shard_path in _existing_shard_paths(output_dir):
    with np.load(shard_path) as shard:
      sample_count += int(shard["actions"].shape[0])
  return sample_count


def _next_shard_index(output_dir: Path) -> int:
  shard_paths = _existing_shard_paths(output_dir)
  if not shard_paths:
    return 0

  return max(int(path.stem.split("_")[-1]) for path in shard_paths) + 1


def _build_manifest(
  policy: TeacherPolicy,
  output_dir: Path,
  episodes_requested_this_run: int,
  total_episodes_target: int,
  episodic_returns: list[float],
  shard_size: int,
  append: bool,
  seed: int,
  complete: bool,
  current_episode: int,
  next_shard_index: int,
) -> dict:
  sample_count = _count_samples(output_dir)
  return {
    "teacher_repo_id": DEFAULT_REPO_ID,
    "teacher_model_path": str(policy.model_path),
    "action_size": policy.action_size,
    "append_mode": append,
    "seed": seed,
    "episodes_requested_this_run": episodes_requested_this_run,
    "target_total_episodes": total_episodes_target,
    "num_episodes": current_episode,
    "episodes_added_this_run": max(0, current_episode - (total_episodes_target - episodes_requested_this_run)),
    "num_samples": sample_count,
    "episodic_returns": episodic_returns,
    "average_reward": float(np.mean(episodic_returns)) if episodic_returns else 0.0,
    "std_reward": float(np.std(episodic_returns)) if episodic_returns else 0.0,
    "shard_count": len(_existing_shard_paths(output_dir)),
    "next_shard_index": next_shard_index,
    "shard_size": shard_size,
    "collection_complete": complete,
    "updated_at": _timestamp(),
  }


def _write_manifest(output_dir: Path, manifest: dict) -> None:
  with _manifest_path(output_dir).open("w", encoding="utf-8") as file:
    json.dump(manifest, file, indent=2)


def _flush_shard(
  output_dir: Path,
  shard_index: int,
  observations: list[np.ndarray],
  actions: list[int],
  rewards: list[float],
  dones: list[bool],
  episode_ids: list[int],
  logits: list[np.ndarray],
) -> int:
  if not observations:
    return shard_index

  shard_path = output_dir / f"teacher_shard_{shard_index:05d}.npz"
  temp_path = Path(f"{shard_path}.tmp")
  with temp_path.open("wb") as file:
    np.savez_compressed(
      file,
      observations=np.asarray(observations, dtype=np.uint8),
      actions=np.asarray(actions, dtype=np.int32),
      rewards=np.asarray(rewards, dtype=np.float32),
      dones=np.asarray(dones, dtype=np.bool_),
      episode_ids=np.asarray(episode_ids, dtype=np.int32),
      logits=np.asarray(logits, dtype=np.float32),
    )
  temp_path.replace(shard_path)
  return shard_index + 1


def collect_trajectories(
  policy: TeacherPolicy,
  output_dir: Path,
  target_episodes: int,
  seed: int,
  shard_size: int,
  append: bool = False,
) -> dict:
  output_dir.mkdir(parents=True, exist_ok=True)
  existing_manifest = _load_existing_manifest(output_dir) if append else None
  if existing_manifest is None and append and _existing_shard_paths(output_dir):
    existing_manifest = _recover_existing_progress(output_dir)

  if not append and _existing_shard_paths(output_dir):
    raise FileExistsError(
      f"{output_dir} already contains trajectory shards. Use --append to continue collecting into the existing dataset."
    )

  starting_episode = int(existing_manifest.get("num_episodes", 0)) if existing_manifest is not None else 0
  episodes_requested_this_run = int(target_episodes)
  target_total_episodes = starting_episode + episodes_requested_this_run
  episodic_returns = list(existing_manifest.get("episodic_returns", [])) if existing_manifest is not None else []

  env = build_env(seed=seed + starting_episode)
  observations_buffer: list[np.ndarray] = []
  actions_buffer: list[int] = []
  rewards_buffer: list[float] = []
  dones_buffer: list[bool] = []
  episode_ids_buffer: list[int] = []
  logits_buffer: list[np.ndarray] = []
  shard_index = _next_shard_index(output_dir)
  current_return = 0.0
  current_episode = starting_episode

  try:
    observation, _ = env.reset()
    while current_episode < target_total_episodes:
      observation = np.asarray(observation)
      logits = policy.predict_logits(observation)
      action = int(np.argmax(logits[0]))
      next_observation, reward, terminated, truncated, _ = env.step(np.asarray([action], dtype=np.int32))
      done = bool(terminated[0]) or bool(truncated[0])

      observations_buffer.append(observation[0].copy())
      actions_buffer.append(action)
      rewards_buffer.append(float(reward[0]))
      dones_buffer.append(done)
      episode_ids_buffer.append(current_episode)
      logits_buffer.append(logits[0].copy())

      current_return += float(reward[0])
      observation = next_observation

      if len(observations_buffer) >= shard_size:
        shard_index = _flush_shard(
          output_dir,
          shard_index,
          observations_buffer,
          actions_buffer,
          rewards_buffer,
          dones_buffer,
          episode_ids_buffer,
          logits_buffer,
        )
        observations_buffer.clear()
        actions_buffer.clear()
        rewards_buffer.clear()
        dones_buffer.clear()
        episode_ids_buffer.clear()
        logits_buffer.clear()
        _write_manifest(
          output_dir,
          _build_manifest(
            policy=policy,
            output_dir=output_dir,
            episodes_requested_this_run=episodes_requested_this_run,
            total_episodes_target=target_total_episodes,
            episodic_returns=episodic_returns,
            shard_size=shard_size,
            append=append,
            seed=seed,
            complete=False,
            current_episode=current_episode,
            next_shard_index=shard_index,
          ),
        )

      if done:
        episodic_returns.append(current_return)
        current_return = 0.0
        current_episode += 1
        _write_manifest(
          output_dir,
          _build_manifest(
            policy=policy,
            output_dir=output_dir,
            episodes_requested_this_run=episodes_requested_this_run,
            total_episodes_target=target_total_episodes,
            episodic_returns=episodic_returns,
            shard_size=shard_size,
            append=append,
            seed=seed,
            complete=False,
            current_episode=current_episode,
            next_shard_index=shard_index,
          ),
        )
        observation, _ = env.reset()
  finally:
    env.close()

  shard_index = _flush_shard(
    output_dir,
    shard_index,
    observations_buffer,
    actions_buffer,
    rewards_buffer,
    dones_buffer,
    episode_ids_buffer,
    logits_buffer,
  )
  metadata = _build_manifest(
    policy=policy,
    output_dir=output_dir,
    episodes_requested_this_run=episodes_requested_this_run,
    total_episodes_target=target_total_episodes,
    episodic_returns=episodic_returns,
    shard_size=shard_size,
    append=append,
    seed=seed,
    complete=True,
    current_episode=current_episode,
    next_shard_index=shard_index,
  )
  _write_manifest(output_dir, metadata)
  return metadata


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Download, evaluate, and collect trajectories from the CleanRL Boxing-v5 teacher.")
  subparsers = parser.add_subparsers(dest="command", required=True)

  download_parser = subparsers.add_parser("download", help="Download the teacher checkpoint from Hugging Face.")
  download_parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
  download_parser.add_argument("--filename", default=DEFAULT_FILENAME)
  download_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/teacher_cleanrl"))

  evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the teacher locally.")
  evaluate_parser.add_argument("--model", type=Path, default=Path("artifacts/teacher_cleanrl") / DEFAULT_FILENAME)
  evaluate_parser.add_argument("--episodes", type=int, default=10)
  evaluate_parser.add_argument("--seed", type=int, default=2)
  evaluate_parser.add_argument("--output", type=Path, default=Path("artifacts/teacher_cleanrl") / "evaluation_summary.json")

  collect_parser = subparsers.add_parser("collect", help="Collect Boxing-v5 trajectories from the teacher.")
  collect_parser.add_argument("--model", type=Path, default=Path("artifacts/teacher_cleanrl") / DEFAULT_FILENAME)
  collect_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/teacher_boxing_dataset"))
  collect_parser.add_argument("--episodes", type=int, default=50)
  collect_parser.add_argument("--seed", type=int, default=2)
  collect_parser.add_argument("--shard-size", type=int, default=2048)
  collect_parser.add_argument("--append", action="store_true", help="Append new episodes into an existing trajectory dataset.")

  return parser.parse_args()


def main() -> None:
  args = parse_args()

  if args.command == "download":
    model_path = download_teacher(args.repo_id, args.filename, args.output_dir)
    print(model_path)
    return

  if args.command == "evaluate":
    policy = load_teacher(args.model, seed=args.seed)
    summary = evaluate_teacher(policy, episodes=args.episodes, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
      json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))
    return

  if args.command == "collect":
    policy = load_teacher(args.model, seed=args.seed)
    summary = collect_trajectories(
      policy=policy,
      output_dir=args.output_dir,
      target_episodes=args.episodes,
      seed=args.seed,
      shard_size=args.shard_size,
      append=args.append,
    )
    print(json.dumps(summary, indent=2))
    return

  raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
  main()
