import json
import os
import numpy as np
import matplotlib
from collections import defaultdict

from exploration.exploration import ExplorationMethod
from replay_memory.replay_memory import ReplayMemory

try:
  matplotlib.use(os.environ.get("MATPLOTLIB_BACKEND", "Agg"))
except ImportError:
  matplotlib.use("Agg")

import matplotlib.pyplot as plt


class ExponentialMovingAverage:
  def __init__(self, alpha: float):
    self.alpha = alpha
    self.value = None

  def update(self, reward: float) -> float:
    if self.value is None:
      self.value = reward
    else:
      self.value = (1 - self.alpha) * self.value + self.alpha * reward
    return self.value


class Agent:
  """
  Abstrakte Basisklasse für Reinforcement-Learning-Agenten.

  Diese Klasse definiert die grundlegende Struktur und Schnittstellen,
  die konkrete RL-Agenten wie Q-Learning, DQN oder SARSA implementieren müssen.
  """

  def __init__(self, action_size: int, exploration: ExplorationMethod,
               memory: ReplayMemory, batch_size: int = 32,
               learning_rate: float = 0.01,
               discount_factor: float = 0.99):
    """
    Initialisiert den Agenten mit den gegebenen Hyperparametern und Komponenten.

    Args:
        action_size (int): Anzahl der möglichen Aktionen im Environment.
        exploration (ExplorationMethod): Strategie zur Aktionsauswahl (z.B. Epsilon-Greedy).
        memory (MemoryReplay): Replay-Memory zur Speicherung von Erfahrungen.
        batch_size (int): Größe des Trainingssamples welches bei Training gelernt wird.
        learning_rate (float, optional): Lernrate (α) für die Wertaktualisierung. Default = 0.01.
        discount_factor (float, optional): Diskontfaktor (γ) für zukünftige Belohnungen. Default = 0.99.
    """
    self._action_size = action_size
    self._exploration = exploration
    self._memory = memory
    self._batch_size = batch_size
    self._learning_rate = learning_rate
    self._discount_factor = discount_factor
    self._rewards = []
    self._reward_ema = ExponentialMovingAverage(alpha=0.05)
    self._episode_reward = 0.0

  def act(self, state: np.ndarray | int, train: bool = True) -> int:
    """
    Wählt eine Aktion basierend auf dem aktuellen Zustand aus.

    Args:
        state (np.ndarray | int): Aktueller Zustand des Environments.
        train (bool, optional): Falls True, wird Exploration verwendet; andernfalls wird greedy gehandelt.

    Returns:
        int: Die gewählte Aktion (Index).
    """
    raise NotImplementedError

  def remember(self, state: np.ndarray | int, action: int, reward: float, next_state: np.ndarray, done: bool):
    """
    Speichert eine Erfahrung (Transition) im Replay-Memory.

    Args:
        state (np.ndarray | int): Ausgangszustand vor der Aktion.
        action (int): Ausgeführte Aktion.
        reward (float): Erhaltene Belohnung.
        next_state (np.ndarray): Nachfolgender Zustand nach der Aktion.
        done (bool): True, wenn die Episode beendet wurde.
    """
    # TODO: "if done": sieht fishy aus: überprüfen
    # self._reward_ema.update(reward)
    # if done:
    #   self._rewards.append(self._reward_ema.value)

    self._episode_reward += reward
    if done:
      self._rewards.append(self._reward_ema.update(self._episode_reward))
      self._episode_reward = 0.0

  @property
  def get_actual_ema_reward_value(self):
    return self._reward_ema.value

  def save_checkpoint(self, path: str) -> None:
    """Speichert den allgemeinen Agent-Zustand (Rewards, EMA, Epsilons)."""
    os.makedirs(path, exist_ok=True)
    state = {
      "rewards": self._rewards,
      "ema_value": self._reward_ema.value,
      "epsilons": self._epsilons if hasattr(self, "_epsilons") else [],
    }
    with open(os.path.join(path, "agent_state.json"), "w") as f:
      json.dump(state, f)

  def load_checkpoint(self, path: str) -> None:
    """Lädt den allgemeinen Agent-Zustand."""
    with open(os.path.join(path, "agent_state.json")) as f:
      state = json.load(f)
    self._rewards = state["rewards"]
    self._reward_ema.value = state["ema_value"]
    if hasattr(self, "_epsilons"):
      self._epsilons = state.get("epsilons", [])

  def train(self) -> None:
    """
    Führt einen Trainingsschritt aus, indem eine Stichprobe aus dem Replay-Memory entnommen wird
    und die Q-Werte anhand der Belohnungen aktualisiert werden.
    """
    raise NotImplementedError

  def after_episode(self) -> None:
    """
    Wird nach jeder Episode aufgerufen, um z. B. den Epsilon-Wert zu reduzieren
    oder interne Zustände zurückzusetzen.
    """
    raise NotImplementedError

  def plot_reward(self) -> None:
    epsilons = self._epsilons if hasattr(self, "_epsilons") and self._epsilons else None
    n_plots = 2 if epsilons else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
      axes = [axes]

    ax_rew = axes[0]
    ax_rew.plot(self._rewards)
    ax_rew.set_title("Reward pro Episode")
    ax_rew.set_xlabel("Episode")
    ax_rew.set_ylabel("Reward (EMA)")
    if self._rewards:
      ax_rew.text(0.98, 0.95, f"R={self._rewards[-1]:.2f}",
                  transform=ax_rew.transAxes, ha="right", va="top")

    if epsilons:
      ax_eps = axes[1]
      ax_eps.plot(epsilons)
      ax_eps.set_title("ε über Episoden")
      ax_eps.set_xlabel("Episode")
      ax_eps.set_ylabel("ε")
      ax_eps.text(0.98, 0.95, f"ε={epsilons[-1]:.3f}",
                  transform=ax_eps.transAxes, ha="right", va="top")

    fig.tight_layout()
    plt.show()

  def _build_policy_grid(self, grid_shape: tuple[int, int]) -> np.ndarray | None:
    return None

  def plot_policy(self, epsilon: list[float], grid_shape: tuple[int, int] = (4, 4)) -> None:
    """
    Zeichnet NACH dem Training:
      - Policy als Pfeil-Grid (nur wenn _build_policy_grid implementiert)
      - Epsilon-Verlauf
      - Reward-Verlauf

    Blocking: Fenster bleibt offen.
    """

    # ---------- Figure ----------
    fig, (ax_policy, ax_eps, ax_rew) = plt.subplots(1, 3, figsize=(12, 4))

    # ---------- Policy ----------
    policy_grid = self._build_policy_grid(grid_shape)
    if policy_grid is not None:
      arrow_map = {0: '←', 1: '↓', 2: '→', 3: '↑', -1: '•'}
      ax_policy.imshow(policy_grid, cmap="viridis", interpolation="nearest")
      ax_policy.set_title("Policy (beste Aktion)")
      ax_policy.set_xticks([])
      ax_policy.set_yticks([])
      for r in range(grid_shape[0]):
        for c in range(grid_shape[1]):
          action = int(policy_grid[r, c])
          symbol = arrow_map.get(action, "•")
          ax_policy.text(c, r, symbol, ha="center", va="center", color="white", fontsize=16)
    else:
      ax_policy.set_title("Policy (nicht verfügbar)")
      ax_policy.text(0.5, 0.5, "kein Q-Table", ha="center", va="center", transform=ax_policy.transAxes)
      ax_policy.set_xticks([])
      ax_policy.set_yticks([])

    # ---------- Epsilon ----------
    ax_eps.plot(epsilon)
    ax_eps.set_title("ε über Episoden")
    ax_eps.set_xlabel("Episode")
    ax_eps.set_ylabel("ε")

    if epsilon:
      ax_eps.text(
        0.98, 0.95, f"ε={epsilon[-1]:.3f}",
        transform=ax_eps.transAxes,
        ha="right", va="top"
      )

    # ---------- Reward ----------
    ax_rew.plot(self._rewards)
    ax_rew.set_title("Reward pro Episode")
    ax_rew.set_xlabel("Episode")
    ax_rew.set_ylabel("Reward")

    if self._rewards:
      ax_rew.text(
        0.98, 0.95, f"R={self._rewards[-1]:.2f}",
        transform=ax_rew.transAxes,
        ha="right", va="top"
      )

    fig.tight_layout()
    plt.show()  # BLOCKING#


class QAgentOffPolicy(Agent):
  def __init__(self, action_size: int, exploration: ExplorationMethod,
               memory: ReplayMemory, batch_size: int = 32,
               learning_rate: float = 0.01,
               discount_factor: float = 0.99):
    super().__init__(action_size, exploration, memory, batch_size,
                     learning_rate, discount_factor)
    self._q_table: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(action_size))
    self._epsilons = []

  def act(self, state: np.ndarray | int, train: bool = True) -> int:
    if isinstance(state, int):
      state_tuple = state
    else:
      state_tuple = tuple(state)
    if not train:
      return np.argmax(self._q_table[state_tuple])
    else:
      return self._exploration.next_action(self._q_table[state_tuple])

  def remember(self, state: np.ndarray | int, action: int,
               reward: float, next_state: np.ndarray, done: bool) -> None:
    super().remember(state=state, action=action, next_state=next_state, reward=reward, done=done)
    self._memory.append([state, action, next_state, reward, done])

  def after_episode(self) -> None:
    self._epsilons.append(self._exploration.epsilon)
    self._exploration.after_episode(self._reward_ema.value)

  def train(self) -> None:
    mini_batch = self._memory.sample(self._batch_size).experiences
    for state, action, next_state, reward, done in mini_batch:
      target = reward if done else reward + self._discount_factor * np.max(self._q_table[next_state])
      self._q_table[state][action] += self._learning_rate * (target - self._q_table[state][action])

  def _build_policy_grid(self, grid_shape: tuple[int, int]) -> np.ndarray | None:
    policy_grid = np.full(grid_shape, -1)
    for state, actions in self._q_table.items():
      best_action = int(np.argmax(actions))
      row = state // grid_shape[1]
      col = state % grid_shape[1]
      if 0 <= row < grid_shape[0] and 0 <= col < grid_shape[1]:
        policy_grid[row, col] = best_action
    return policy_grid

  def plot_policy_easy(self, grid_shape: tuple[int, int] = (4, 4)) -> None:
    super().plot_policy(self._epsilons, grid_shape=grid_shape)










# class _DQNAgentOffPolicy(Agent):
#   def __init__(self, action_size: int, exploration: ExplorationMethod,
#                memory: ReplayMemory, state_size: list[int], batch_size: int = 32,
#                learning_rate: float = 0.01,
#                discount_factor: float = 0.99):
#     super().__init__(action_size, exploration, memory, batch_size,
#                      learning_rate, discount_factor)
#     self._state_size = state_size
#     self._model = self._build_model()
#     self._epsilons = []
#
#   def _build_model(self) -> keras.Model:
#     model = keras.Sequential([
#       keras.layers.Flatten(input_shape=self._state_size),
#       keras.layers.Dense(32, activation="relu"),
#       keras.layers.Dense(self._action_size, activation="linear")
#     ])
#     model.compile(loss="MSE", optimizer=keras.optimizers.Adam(learning_rate=self._learning_rate))
#     print(model.summary())
#     return model
#
#   def _state_to_batch(self, state) -> np.ndarray:
#     return np.array([[np.array(e) for e in state]])

  # def act(self, state: np.ndarray | int, train: bool = True) -> int:
  #   state = self._state_to_batch(state)
  #   self._model.predict(state)



# if __name__ == "__main__":
#   agent = DQNAgentOffPolicy(action_size=3, exploration=None,
#                 memory=None, state_size=[8, 8])

