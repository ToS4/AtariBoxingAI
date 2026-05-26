import json
import os
from typing import Callable
import tensorflow as tf
import keras
import numpy as np

from agent.agent import Agent
from exploration.exploration import ExplorationMethod
from replay_memory.replay_memory import ReplayMemory


class DDQN_Agent_Off_Policy(Agent):
  def __init__(self, model_fn: Callable[[], keras.Model], action_size: int, exploration: ExplorationMethod,
               memory: ReplayMemory, batch_size: int = 32,
               learning_rate: float = 0.01,
               discount_factor: float = 0.99,
               target_update_freq: int = 10):
    super().__init__(action_size, exploration, memory, batch_size,
                     learning_rate, discount_factor)

    self._model = model_fn()         # Online-Netzwerk
    self._target_model = model_fn()  # Target-Netzwerk
    self._update_target_model()

    self._epsilons = []
    self._target_update_freq = target_update_freq
    self._train_counter = 0

  def _update_target_model(self) -> None:
    self._target_model.set_weights(self._model.get_weights())

  def _state_to_numpy(self, state) -> np.ndarray:
    return np.array([[np.array(e) for e in state]])

  def act(self, state: np.ndarray | int, train: bool = True) -> int:
    state = self._state_to_numpy(state)
    q_values = self._model(state, training=train).numpy()[0]
    if not train:
      return int(np.argmax(q_values))
    return self._exploration.next_action(q_values)

  def remember(self, state: np.ndarray | int, action: int,
               reward: float, next_state: np.ndarray | int, done: bool) -> None:
    state = self._state_to_numpy(state)
    next_state = self._state_to_numpy(next_state)

    super().remember(state=state, action=action, next_state=next_state, reward=reward, done=done)
    self._memory.append([state, action, next_state, reward, done])

  def after_episode(self) -> None:
    if hasattr(self._exploration, "epsilon"):
      self._epsilons.append(self._exploration.epsilon)
    self._exploration.after_episode(self._reward_ema.value)

  def train(self) -> None:
    batch = self._memory.sample(self._batch_size)
    mini_batch = batch.experiences

    if len(mini_batch) < self._batch_size:
      return

    states = np.vstack([entry[0] for entry in mini_batch])
    actions = [entry[1] for entry in mini_batch]
    next_states = np.vstack([entry[2] for entry in mini_batch])
    rewards = [entry[3] for entry in mini_batch]
    dones = [entry[4] for entry in mini_batch]

    q_values = self._model(states, training=False).numpy()
    # DDQN: Online-Netz wählt Aktion, Target-Netz bewertet sie
    q_next_online = self._model(next_states, training=False).numpy()
    q_next_target = self._target_model(next_states, training=False).numpy()
    best_next_actions = np.argmax(q_next_online, axis=1)

    td_errors = []

    for i in range(self._batch_size):
      old_value = q_values[i][actions[i]]

      if dones[i]:
        target = rewards[i]
      else:
        target = rewards[i] + self._discount_factor * q_next_target[i][best_next_actions[i]]

      td_errors.append(target - old_value)
      q_values[i][actions[i]] = target

    if batch.indices is not None:
      self._memory.update_priorities(batch.indices, np.array(td_errors))

    if batch.weights is not None:
      self._model.train_on_batch(states, q_values, sample_weight=batch.weights)
    else:
      self._model.train_on_batch(states, q_values)

    self._train_counter += 1
    if self._train_counter % self._target_update_freq == 0:
      self._update_target_model()

  def plot_policy_easy(self, grid_shape: tuple[int, int] = (4, 4)) -> None:
    super().plot_policy(self._epsilons, grid_shape=grid_shape)

  def save_checkpoint(self, path: str) -> None:
    super().save_checkpoint(path)
    self._model.save(os.path.join(path, "model.keras"))
    self._target_model.save(os.path.join(path, "target_model.keras"))
    with open(os.path.join(path, "train_state.json"), "w") as f:
      json.dump({
        "train_counter": self._train_counter,
        "epsilon": self._exploration.epsilon if hasattr(self._exploration, "epsilon") else None,
      }, f)

  def load_checkpoint(self, path: str) -> None:
    super().load_checkpoint(path)
    self._model = keras.models.load_model(os.path.join(path, "model.keras"))
    self._target_model = keras.models.load_model(os.path.join(path, "target_model.keras"))
    with open(os.path.join(path, "train_state.json")) as f:
      state = json.load(f)
    self._train_counter = state["train_counter"]
    if hasattr(self._exploration, "epsilon") and state.get("epsilon") is not None:
      self._exploration.epsilon = state["epsilon"]

  def save_model(self, filepath: str) -> None:
    self._model.save(filepath)

  def load_model(self, filepath: str) -> None:
    self._model = keras.models.load_model(filepath)
    self._target_model = keras.models.clone_model(self._model)
    self._target_model.set_weights(self._model.get_weights())
    self._target_model.compile(
      loss="MSE",
      optimizer=tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
    )


if __name__ == "__main__":
  agent = DDQN_Agent_Off_Policy(
    state_size=[8, 8],
    action_size=4,
    exploration=None,
    memory=None
  )