import abc
import random
from abc import abstractmethod
import numpy as np


class ExplorationMethod(abc.ABC):
  """
  Abstrakte Basisklasse für Explorationsstrategien im Reinforcement Learning.

  Eine Explorationsstrategie bestimmt, wie ein Agent seine Aktionen auswählt,
  wenn er zwischen bekannten, optimalen Entscheidungen (Exploitation) und
  neuen, unsicheren Entscheidungen (Exploration) abwägen muss.
  Typische Implementierungen sind z. B. Epsilon-Greedy oder Softmax.

  Parameters
  ----------
  action_size : int
      Anzahl der möglichen Aktionen, die der Agent ausführen kann.
  """

  def __init__(self, action_size: int):
    """
    Initialisiert die Explorationsmethode mit der Anzahl möglicher Aktionen.

    Parameters
    ----------
    action_size : int
        Anzahl der möglichen Aktionen.
    """
    self._action_size = action_size

  @abstractmethod
  def next_action(self, q_values: np.ndarray) -> int:
    """
    Wählt die nächste Aktion basierend auf den gegebenen Q-Werten.

    Die Implementierung legt fest, ob eine explorative (zufällige) oder
    eine exploitative (argmax) Aktion ausgeführt wird.

    Parameters
    ----------
    q_values : np.ndarray
        Array mit den geschätzten Q-Werten für jede mögliche Aktion.

    Returns
    -------
    int
        Der Index der gewählten Aktion (zwischen 0 und action_size-1).
    """
    raise NotImplementedError()

  @abstractmethod
  def after_episode(self, ema_reward: float | None = None) -> None:
    """
    Führt Anpassungen nach einer abgeschlossenen Episode durch.

    Beispiele:
    - Reduktion des Epsilon-Werts bei Epsilon-Greedy
    - Anpassung der Temperatur bei Softmax
    - Logging oder andere Updates

    Returns
    -------
    None
    """
    raise NotImplementedError()

  @property
  def action_size(self) -> int:
    """
    Gibt die Anzahl der möglichen Aktionen zurück.

    Returns
    -------
    int
        Anzahl der Aktionen (Action-Space-Größe).
    """
    return self._action_size


class EGreedy(ExplorationMethod):
  def __init__(self, epsilon_start: float, epsilon_end: float, episodes_count: int, action_size: int):
    super().__init__(action_size=action_size)
    self._episodes_count = episodes_count
    self._epsilon_start = epsilon_start
    self._epsilon = epsilon_start
    self._epsilon_end = epsilon_end
    self._epsilon_decay = (epsilon_start - epsilon_end) / episodes_count

  def next_action(self, q_values: np.ndarray) -> int:
    random_value = random.random()
    if random_value < self._epsilon:
      return random.randint(0, self.action_size-1)
    return int(np.argmax(q_values))

  @property
  def epsilon(self) -> float:
    return self._epsilon

  @epsilon.setter
  def epsilon(self, value: float) -> None:
    self._epsilon = value

  def after_episode(self, ema_reward: float | None = None) -> None:
    self._epsilon = max(self._epsilon - self._epsilon_decay, self._epsilon_end)


class EGreedyExp(ExplorationMethod):
  """Epsilon-Greedy mit exponentiellem Decay: ε *= decay_rate pro Episode.
  decay_rate wird automatisch so berechnet, dass ε nach episodes_count Episoden epsilon_end erreicht.
  """

  def __init__(self, epsilon_start: float, epsilon_end: float, episodes_count: int, action_size: int):
    super().__init__(action_size=action_size)
    self._epsilon = epsilon_start
    self._epsilon_end = epsilon_end
    self._decay_rate = (epsilon_end / epsilon_start) ** (1.0 / episodes_count)

  def next_action(self, q_values: np.ndarray) -> int:
    if random.random() < self._epsilon:
      return random.randint(0, self.action_size - 1)
    return int(np.argmax(q_values))

  @property
  def epsilon(self) -> float:
    return self._epsilon

  @epsilon.setter
  def epsilon(self, value: float) -> None:
    self._epsilon = value

  def after_episode(self, ema_reward: float | None = None) -> None:
    self._epsilon = max(self._epsilon * self._decay_rate, self._epsilon_end)


class AdaptiveEGreedyExp(ExplorationMethod):
  """Epsilon-Greedy mit adaptivem exponentiellem Decay.
  Decay verlangsamt sich wenn der EMA-Reward stagniert, läuft normal wenn er sich verbessert.
  slow_factor (0..1): Anteil des normalen Decays bei Stagnation (z.B. 0.1 = 10x langsamer).
  """

  def __init__(self, epsilon_start: float, epsilon_end: float, episodes_count: int,
               action_size: int, slow_factor: float = 0.1):
    super().__init__(action_size=action_size)
    self._epsilon = epsilon_start
    self._epsilon_end = epsilon_end
    self._decay_rate = (epsilon_end / epsilon_start) ** (1.0 / episodes_count)
    self._slow_factor = slow_factor
    self._prev_ema: float | None = None

  def next_action(self, q_values: np.ndarray) -> int:
    if random.random() < self._epsilon:
      return random.randint(0, self.action_size - 1)
    return int(np.argmax(q_values))

  @property
  def epsilon(self) -> float:
    return self._epsilon

  @epsilon.setter
  def epsilon(self, value: float) -> None:
    self._epsilon = value

  def after_episode(self, ema_reward: float | None = None) -> None:
    improving = ema_reward is not None and self._prev_ema is not None and ema_reward > self._prev_ema
    rate = self._decay_rate if improving else self._decay_rate ** self._slow_factor
    self._epsilon = max(self._epsilon * rate, self._epsilon_end)
    self._prev_ema = ema_reward