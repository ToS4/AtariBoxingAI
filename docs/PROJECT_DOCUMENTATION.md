# Projektdokumentation - MohiBoxer

## Ziel

Ziel des Projekts war es, einen trainierten Keras-Agenten für Atari Boxing zu erstellen. Der Agent soll direkt geladen werden können und gegen den eingebauten Atari-Gegner spielbar sein.

Finales Modell:

```text
models/mohi_boxer_best.h5
```

## Agent

Der finale Agent ist ein Keras-Modell. Er bekommt den aktuellen Spielzustand als vorverarbeitetes Bild und gibt für jede mögliche Aktion einen Wert aus.

Beim Spielen wird keine zufällige Aktion mehr gewählt. Der Agent nimmt einfach die Aktion mit dem höchsten Modellwert:

```text
action = argmax(model(observation))
```

Im Code gibt es auch DQN- und DDQN-Agenten. Diese waren der ursprüngliche RL-Ansatz. Für das beste finale Modell war aber Teacher-Student-Training viel schneller und erfolgreicher.

## Replay Memory

Beim ursprünglichen DQN/DDQN-Ansatz gibt es Replay Memory.

Replay Memory bedeutet:

- Erfahrungen werden gespeichert.
- Später werden zufällige Beispiele daraus wieder zum Lernen benutzt.
- Dadurch lernt der Agent stabiler.

Im Projekt gibt es:

- normales Replay Memory
- Prioritized Replay Memory
- Replay-Größe: `50,000`

Beim finalen Modell war Replay Memory aber nicht der wichtigste Teil. Das finale Modell wurde hauptsächlich mit einer großen Teacher-Datenmenge trainiert:

- `20,202` gelabelte Episoden
- `8,966,962` Trainingsbeispiele

Diese Datenmenge funktioniert ähnlich wie ein großer Offline-Speicher mit guten Beispielen.

## Exploration

Beim ursprünglichen DQN/DDQN wurde Epsilon-Greedy verwendet.

Das bedeutet:

- Am Anfang probiert der Agent viele zufällige Aktionen.
- Später nimmt er immer öfter die beste bekannte Aktion.

Hyperparameter:

- Epsilon Start: `1.0`
- Epsilon Ende: `0.1`

Beim finalen Modell gibt es in der Evaluation keine Exploration mehr. Das Modell spielt deterministisch und wählt immer die Aktion mit dem höchsten Wert.

## Reward-Funktion

Es wurde der normale Reward von Atari Boxing verwendet.

Der Agent bekommt Punkte, wenn er Schläge trifft. Es wurde kein eigener zusätzlicher Reward gebaut.

Für das finale Training wurden Sticky Actions aktiviert:

```text
repeat_action_probability = 0.25
```

Das bedeutet, dass manchmal die vorherige Aktion wiederholt wird. Dadurch muss der Agent robuster spielen.

## Neuronales Netz

Das finale Modell hat:

- Input: `(4, 84, 84)`
- Output: `18`
- Parameter: `1,093,858`

Aufbau:

1. Input: 4 gestapelte graue Atari-Bilder
2. Umordnung in channels-last Format
3. Pixel werden durch `255` geteilt
4. mehrere Convolutional Layers mit Residual Blocks
5. Flatten
6. Dense Layer mit `256` ReLU-Neuronen
7. Output Layer mit `18` Werten

Warum CNN?

Der Agent bekommt Bilder. CNNs sind gut für Bilder, weil sie Positionen, Formen und Bewegungen besser erkennen können als ein normales Dense-Netz.

## Observation

Das Atari-Spiel liefert ursprünglich RGB-Bilder.

Vorverarbeitung:

- Bild wird in Graustufen umgewandelt
- Bild wird auf `84x84` verkleinert
- mehrere Frames werden übersprungen
- die letzten 4 Frames werden gestapelt
- Pixel bleiben als `uint8`

Finaler Input:

```text
(4, 84, 84)
```

Warum 4 Frames?

Ein einzelnes Bild zeigt keine Bewegung. Mit 4 Bildern kann das Modell erkennen, ob sich ein Boxer nach vorne, hinten oder zur Seite bewegt.

## Aktionen

Boxing verwendet den vollständigen Atari Action Space:

```text
Discrete(18)
```

Das Modell gibt 18 Werte aus. Jeder Wert steht für eine mögliche Aktion. Die Aktion mit dem höchsten Wert wird ausgeführt.

## Training

Das finale Training war Teacher-Student-Training.

Ablauf:

1. Ein starker Teacher-Agent spielt Boxing.
2. Seine Aktionen werden gespeichert.
3. Unser Keras-Student lernt diese Aktionen nachzumachen.
4. Danach spielt der Student selbst.
5. Der Teacher beschriftet schwierige Situationen des Students.
6. Das Modell wird erneut trainiert und evaluiert.
7. Das beste Modell wird gespeichert.

Finaler Trainingsstand:

- `337` abgeschlossene Trainingsrunden
- `20,202` gelabelte Episoden
- `8,966,962` Trainingsbeispiele

Bestes 10-Spiele-Ergebnis:

```text
100, 100, 100, 100, 100, 100, 100, 100, 100, 100
```

Finale 5-Spiele-Evaluation:

```text
98, 100, 100, 100, 98
```

Durchschnitt:

```text
99.2
```

## Beweisdateien

- Finales Modell: `models/mohi_boxer_best.h5`
- Reward-Kurve: `artifacts/training/training_reward_curve.png`
- Evaluation: `artifacts/evaluation/final_5_game_summary.json`
- Videos: `artifacts/evaluation/videos/`
- Modellzusammenfassung: `artifacts/training/model_summary.txt`
