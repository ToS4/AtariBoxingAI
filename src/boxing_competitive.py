from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CV2_IMPORT_ERROR: Exception | None = None
try:
  import cv2
except ImportError as exc:  # pragma: no cover - handled at runtime
  cv2 = None
  _CV2_IMPORT_ERROR = exc

_KERAS_IMPORT_ERROR: Exception | None = None
try:
  import keras
except ImportError as exc:  # pragma: no cover - handled at runtime
  keras = None
  _KERAS_IMPORT_ERROR = exc

_PZ_IMPORT_ERROR: Exception | None = None
try:
  from pettingzoo.atari import boxing_v2
except ImportError as exc:  # pragma: no cover - handled at runtime
  boxing_v2 = None
  _PZ_IMPORT_ERROR = exc

_MA_ALE_IMPORT_ERROR: Exception | None = None
try:
  import multi_agent_ale_py
except ImportError as exc:  # pragma: no cover - handled at runtime
  multi_agent_ale_py = None
  _MA_ALE_IMPORT_ERROR = exc


@dataclass(slots=True)
class CompetitiveConfig:
  screen_size: int = 84
  stack_size: int = 4
  max_cycles: int = 1_200
  full_action_space: bool = True
  episodes: int = 5
  seed: int = 42


def _require_dependencies() -> None:
  if cv2 is None:
    raise RuntimeError(
      "opencv-python is required for competitive preprocessing. Install the packages from requirements.txt first."
    ) from _CV2_IMPORT_ERROR

  if keras is None:
    raise RuntimeError(
      "Keras is required to load the saved model. Install the packages from requirements.txt first."
    ) from _KERAS_IMPORT_ERROR

  if boxing_v2 is None:
    raise RuntimeError(
      "PettingZoo Atari support is required for competitive Boxing. Install the optional tournament packages first."
    ) from _PZ_IMPORT_ERROR

  if multi_agent_ale_py is None:
    raise RuntimeError(
      "multi_agent_ale_py is required for competitive Boxing. Install the tournament packages in Linux / WSL."
    ) from _MA_ALE_IMPORT_ERROR


def resolve_auto_rom_install_path() -> str | None:
  if multi_agent_ale_py is None:
    return None

  package_dir = Path(multi_agent_ale_py.__file__).resolve().parent
  if (package_dir / "roms" / "boxing.bin").exists():
    return str(package_dir)
  return None


def preprocess_frame(frame: np.ndarray, screen_size: int) -> np.ndarray:
  gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  resized = cv2.resize(gray, (screen_size, screen_size), interpolation=cv2.INTER_AREA)
  return resized.astype(np.uint8)


class FrameStacker:
  def __init__(self, stack_size: int):
    self._frames: deque[np.ndarray] = deque(maxlen=stack_size)
    self._stack_size = stack_size

  def reset(self, frame: np.ndarray) -> np.ndarray:
    self._frames.clear()
    for _ in range(self._stack_size):
      self._frames.append(frame)
    return np.stack(list(self._frames), axis=0)

  def update(self, frame: np.ndarray) -> np.ndarray:
    if not self._frames:
      return self.reset(frame)
    self._frames.append(frame)
    return np.stack(list(self._frames), axis=0)


class RandomPolicy:
  def act(self, observation: np.ndarray, action_size: int) -> int:
    del observation
    return int(np.random.randint(action_size))


class KerasPolicy:
  def __init__(self, model_path: Path):
    _require_dependencies()
    self._model = keras.models.load_model(model_path)

  def act(self, observation: np.ndarray, action_size: int) -> int:
    del action_size
    q_values = self._model(np.expand_dims(observation, axis=0), training=False).numpy()[0]
    return int(np.argmax(q_values))


def build_env(config: CompetitiveConfig, render_mode: str | None = None):
  _require_dependencies()
  auto_rom_install_path = resolve_auto_rom_install_path()
  return boxing_v2.parallel_env(
    render_mode=render_mode,
    obs_type="rgb_image",
    full_action_space=config.full_action_space,
    max_cycles=config.max_cycles,
    auto_rom_install_path=auto_rom_install_path,
  )


def run_competitive_match(
  first_policy,
  second_policy,
  config: CompetitiveConfig,
  render_mode: str | None = None,
) -> dict:
  env = build_env(config=config, render_mode=render_mode)
  results: list[dict[str, float]] = []

  for episode in range(config.episodes):
    observations, _ = env.reset(seed=config.seed + episode)
    stackers = {agent: FrameStacker(config.stack_size) for agent in observations}
    processed_observations = {
      agent: stackers[agent].reset(preprocess_frame(frame, config.screen_size))
      for agent, frame in observations.items()
    }
    episode_rewards = {"first_0": 0.0, "second_0": 0.0}

    while env.agents:
      actions = {}
      for agent in env.agents:
        action_size = int(env.action_space(agent).n)
        policy = first_policy if agent == "first_0" else second_policy
        actions[agent] = policy.act(processed_observations[agent], action_size)

      observations, rewards, terminations, truncations, _ = env.step(actions)

      for agent, reward in rewards.items():
        episode_rewards[agent] = episode_rewards.get(agent, 0.0) + float(reward)

      for agent, frame in observations.items():
        processed = preprocess_frame(frame, config.screen_size)
        processed_observations[agent] = stackers.setdefault(agent, FrameStacker(config.stack_size)).update(processed)

      if all(terminations.get(agent, False) or truncations.get(agent, False) for agent in ("first_0", "second_0")):
        break

    results.append(episode_rewards)

  env.close()

  first_rewards = [entry["first_0"] for entry in results]
  second_rewards = [entry["second_0"] for entry in results]
  return {
    "episodes": results,
    "first_average_reward": float(np.mean(first_rewards)) if first_rewards else 0.0,
    "second_average_reward": float(np.mean(second_rewards)) if second_rewards else 0.0,
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run a saved Keras Boxing model in the two-player tournament environment.")
  parser.add_argument("--first-model", type=Path, required=True, help="Model path for first_0.")
  parser.add_argument("--second-model", type=Path, default=None, help="Optional model path for second_0. Defaults to random.")
  parser.add_argument("--episodes", type=int, default=5)
  parser.add_argument("--render-mode", choices=("human",), default=None)
  parser.add_argument("--output", type=Path, default=Path("artifacts/boxing_competitive_summary.json"))
  parser.add_argument("--seed", type=int, default=42)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  config = CompetitiveConfig(episodes=args.episodes, seed=args.seed)

  first_policy = KerasPolicy(args.first_model)
  second_policy = KerasPolicy(args.second_model) if args.second_model is not None else RandomPolicy()
  summary = run_competitive_match(
    first_policy=first_policy,
    second_policy=second_policy,
    config=config,
    render_mode=args.render_mode,
  )

  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
