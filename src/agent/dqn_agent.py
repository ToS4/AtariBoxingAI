import json
import os
from typing import Callable
import tensorflow as tf
import keras
import numpy as np

from agent.agent import Agent
from exploration.exploration import ExplorationMethod
from replay_memory.replay_memory import ReplayMemory

class DQN_Agent_Off_Policy(Agent):
  def __init__(self, model_fn: Callable[[], keras.Model], action_size: int, exploration: ExplorationMethod,
               memory: ReplayMemory, batch_size: int = 32,
               learning_rate: float = 0.01,
               discount_factor: float = 0.99):
    super().__init__(action_size, exploration, memory, batch_size,
                     learning_rate, discount_factor)
    self._model = model_fn()
    self._epsilons = []

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

    all_states = np.vstack([states, next_states])
    all_q = self._model(all_states, training=False).numpy()
    q_values = all_q[:self._batch_size]
    q_next_values = all_q[self._batch_size:]

    td_errors = []
    for i in range(self._batch_size):
      old_value = q_values[i][actions[i]]
      if dones[i]:
        target = rewards[i]
      else:
        target = rewards[i] + self._discount_factor * np.max(q_next_values[i])
      td_errors.append(target - old_value)
      q_values[i][actions[i]] = target

    if batch.indices is not None:
      self._memory.update_priorities(batch.indices, np.array(td_errors))

    if batch.weights is not None:
      self._model.train_on_batch(states, q_values, sample_weight=batch.weights)
    else:
      self._model.train_on_batch(states, q_values)

  def plot_policy_easy(self, grid_shape: tuple[int, int] = (4, 4)) -> None:
    super().plot_policy(self._epsilons, grid_shape=grid_shape)

  def save_checkpoint(self, path: str) -> None:
    super().save_checkpoint(path)
    self._model.save(os.path.join(path, "model.keras"))
    with open(os.path.join(path, "train_state.json"), "w") as f:
      json.dump({"epsilon": self._exploration.epsilon}, f)

  def load_checkpoint(self, path: str) -> None:
    super().load_checkpoint(path)
    self._model = keras.models.load_model(os.path.join(path, "model.keras"))
    with open(os.path.join(path, "train_state.json")) as f:
      state = json.load(f)
    self._exploration.epsilon = state["epsilon"]

  def save_model(self, filepath: str) -> None:
    self._model.save(filepath)

  def load_model(self, filepath: str) -> None:
    self._model = keras.models.load_model(filepath)


if __name__ == "__main__":
  agent = DQN_Agent_Off_Policy(state_size=[8, 8], action_size=4, exploration=None,
                               memory=None)
