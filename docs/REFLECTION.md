# Reflexion

## Was gut funktioniert hat

Am besten funktioniert hat der Wechsel von normalem DQN/DDQN zu Teacher-Student-Training.

Bei DQN/DDQN muss der Agent sehr viel selbst ausprobieren. Das ist bei Atari Boxing schwierig, weil gutes Verhalten nicht sofort offensichtlich ist. Der Agent muss lernen, wie weit er vom Gegner entfernt sein soll, wann er schlagen soll und wie er sich verteidigt.

Mit Teacher-Student-Training war das deutlich schneller. Ein starker Teacher-Agent hat gute Aktionen vorgemacht. Unser Keras-Modell konnte diese Aktionen nachlernen.

Auch die wiederholten Rescue-Runden haben geholfen. Dabei spielt der Student selbst, und der Teacher beschriftet dann die Situationen, in denen der Student landet. Dadurch lernt das Modell nicht nur perfekte Teacher-Situationen, sondern auch Situationen, die durch eigene Fehler entstehen.

Wichtig war außerdem das Training mit Sticky Actions. Bei Sticky Actions wird manchmal die vorherige Aktion wiederholt. Dadurch ist das Spiel realistischer und schwieriger. Frühere Modelle waren ohne Sticky Actions stark, aber mit Sticky Actions schwächer. Das finale Modell wurde deshalb robuster trainiert.

## Was nicht so gut funktioniert hat

Reines DDQN war für die verfügbare Zeit zu langsam. Der Code enthält Replay Memory, Target Network und Epsilon-Greedy Exploration, aber Atari Boxing braucht sehr viele Spiele, bis ein Agent allein durch Ausprobieren gut wird.

Das GPU-Setup war auch schwierig. Auf dem Server gab es moderne RTX-GPUs, aber nicht jede TensorFlow-Version funktionierte direkt damit. Deshalb war CPU-Fallback wichtig. Für die Abgabe ist CPU-Evaluation am sichersten, weil sie zuverlässiger läuft.

Ein weiterer wichtiger Punkt: Das neueste Modell war nicht automatisch das beste Modell. Die letzte Runde hatte einen schlechteren Durchschnitt als ein früher gespeichertes Modell. Deshalb wurde für die Abgabe bewusst das beste gespeicherte Modell verwendet.

## Warum diese Architektur sinnvoll ist

Der Agent bekommt Bilder als Input. Deshalb ist ein Convolutional Neural Network sinnvoll.

Ein CNN kann räumliche Informationen aus Bildern lernen, zum Beispiel:

- wo der eigene Boxer steht
- wo der Gegner steht
- ob ein Schlag möglich ist
- wie sich die Boxer bewegen

Das finale Netz ist nicht extrem groß. Es hat ungefähr `1.09 Millionen` Parameter. Das ist groß genug für Atari-Bilder, aber noch klein genug, um schnell zu trainieren und zu evaluieren.

## Was ich nächstes Mal anders machen würde

Ich würde früher klären, wie genau das Turnier ablaufen wird. Dann könnte man gezielt gegen verschiedene Gegnertypen trainieren und nicht nur gegen den eingebauten Atari-Gegner.

Außerdem würde ich von Anfang an alle Experimente sauberer in eigenen Ordnern speichern. Dann wären Trainingskurven, Modelle, Videos und Konfigurationen noch einfacher zu vergleichen.

Als nächsten technischen Schritt würde ich gegen mehrere verschiedene Agenten trainieren. Das Ziel wäre, nicht nur gegen den Standard-Gegner gut zu sein, sondern auch gegen Mitschüler-Modelle im Turnier.

## Fazit

Das finale Modell ist für die Abgabe geeignet, weil:

- es als Keras `.h5` Modell gespeichert ist
- es direkt über den README-Befehl geladen werden kann
- es eine Reward-Kurve gibt
- es eine 5-Spiele-Evaluation gegen den Standard-Gegner gibt
- es Videos von dieser Evaluation gibt
- die wichtigsten Entscheidungen dokumentiert sind
