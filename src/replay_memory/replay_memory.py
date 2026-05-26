from abc import abstractmethod
from collections import deque
import random

import numpy as np
import abc
from dataclasses import dataclass


@dataclass
class ReplayBatch:
  experiences: list
  indices: np.ndarray | None = None
  weights: np.ndarray | None = None


class ReplayMemory(abc.ABC):
  """
  Abstrakte Basisklasse für Replay-Memory-Puffer.
  """

  def __init__(self, size: int):
    self._size = size

  @abstractmethod
  def __len__(self) -> int:
    raise NotImplementedError()

  @property
  def size(self) -> int:
    return self._size

  @abstractmethod
  def append(self, experience: np.ndarray | list) -> None:
    raise NotImplementedError()

  @abstractmethod
  def sample(self, batch_size: int) -> ReplayBatch:
    """
    Liefert ein Mini-Batch zurück.

    Returns:
        ReplayBatch:
          - experiences: die gezogenen Erfahrungen
          - indices: Positionen im Buffer (wichtig für PER)
          - weights: Importance-Sampling-Gewichte (optional)
    """
    raise NotImplementedError()

  def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
    """
    Standardmäßig keine Aktion.
    ExpReplay braucht keine Prioritäten.
    PrioReplay überschreibt diese Methode.
    """
    pass


class ExpReplay(ReplayMemory):
  def __init__(self, size: int):
    super().__init__(size)
    self._memory = deque(maxlen=size)

  def __len__(self) -> int:
    return len(self._memory)

  def append(self, experience: np.ndarray | list) -> None:
    self._memory.append(experience)

  def sample(self, batch_size: int) -> ReplayBatch:
    batch_size = min(batch_size, len(self._memory))
    experiences = random.sample(self._memory, batch_size)
    return ReplayBatch(experiences=experiences)


class _SumTree:
  """Segment Tree für O(log N) Prioritäts-Summen und Sampling."""

  def __init__(self, capacity: int):
    self._capacity = capacity
    self._tree = np.zeros(2 * capacity, dtype=np.float32)

  def update(self, leaf_idx: int, value: float) -> None:
    idx = leaf_idx + self._capacity
    self._tree[idx] = value
    idx //= 2
    while idx >= 1:
      self._tree[idx] = self._tree[2 * idx] + self._tree[2 * idx + 1]
      idx //= 2

  def get(self, value: float) -> tuple[int, float]:
    """Gibt (leaf_idx, priority) für einen Wert zwischen 0 und total zurück."""
    idx = 1
    while idx < self._capacity:
      left = 2 * idx
      if value <= self._tree[left]:
        idx = left
      else:
        value -= self._tree[left]
        idx = left + 1
    leaf_idx = idx - self._capacity
    return leaf_idx, self._tree[idx]

  @property
  def total(self) -> float:
    return float(self._tree[1])


class _MaxTree:
  """Segment Tree für O(log N) Prioritäts-Maximum."""

  def __init__(self, capacity: int):
    self._capacity = capacity
    self._tree = np.zeros(2 * capacity, dtype=np.float32)

  def update(self, leaf_idx: int, value: float) -> None:
    idx = leaf_idx + self._capacity
    self._tree[idx] = value
    idx //= 2
    while idx >= 1:
      self._tree[idx] = max(self._tree[2 * idx], self._tree[2 * idx + 1])
      idx //= 2

  @property
  def max(self) -> float:
    return float(self._tree[1])


class PrioReplay(ReplayMemory):
  """
  Prioritized Experience Replay (Schaul et al., 2015).
  Nutzt Sum Tree und Max Tree für O(log N) append, sample und update.

  Prioritäten werden als p^alpha gespeichert, damit sample() direkt
  p_i / total als Wahrscheinlichkeit verwenden kann.
  """

  def __init__(self, size: int, alpha: float = 0.6, beta: float = 0.4, epsilon: float = 1e-5):
    super().__init__(size)
    self._memory = [None] * size
    self._sum_tree = _SumTree(size)
    self._max_tree = _MaxTree(size)
    self._alpha = alpha
    self._beta = beta
    self._epsilon = epsilon
    self._position = 0
    self._count = 0

  def __len__(self) -> int:
    return self._count

  def append(self, experience: np.ndarray | list) -> None:
    # Neue Experience bekommt höchste bekannte Priorität (p^alpha)
    max_priority = self._max_tree.max or 1.0
    self._memory[self._position] = experience
    self._sum_tree.update(self._position, max_priority)
    self._max_tree.update(self._position, max_priority)
    self._position = (self._position + 1) % self._size
    self._count = min(self._count + 1, self._size)

  def sample(self, batch_size: int) -> ReplayBatch:
    batch_size = min(batch_size, self._count)
    total = self._sum_tree.total

    # Stratified sampling: Buffer in batch_size gleiche Segmente aufteilen
    segment = total / batch_size
    indices = np.empty(batch_size, dtype=np.int64)
    priorities = np.empty(batch_size, dtype=np.float32)

    for i in range(batch_size):
      value = random.uniform(segment * i, segment * (i + 1))
      idx, priority = self._sum_tree.get(value)
      indices[i] = idx
      priorities[i] = priority

    probabilities = priorities / total
    weights = (self._count * probabilities) ** (-self._beta)
    weights /= weights.max()

    experiences = [self._memory[i] for i in indices]
    return ReplayBatch(
      experiences=experiences,
      indices=indices,
      weights=weights.astype(np.float32),
    )

  def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
    for idx, td_error in zip(indices, td_errors):
      priority = (abs(float(td_error)) + self._epsilon) ** self._alpha
      self._sum_tree.update(idx, priority)
      self._max_tree.update(idx, priority)
