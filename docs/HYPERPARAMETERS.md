# Hyperparameter

## Finaler Abgabe-Agent

| Parameter | Wert |
| --- | --- |
| Environment | `ALE/Boxing-v5` |
| Finales Modell | `models/mohi_boxer_best.h5` |
| Modellformat | Keras `.h5` plus `.keras` Backup |
| Finaler Trainingsansatz | Teacher-Student Distillation mit DAgger-ähnlichen Rescue-Runden |
| Aktion beim Spielen | Greedy `argmax(logits)` |
| Teacher-Familie | CleanRL PPO Atari Boxing Teacher |
| Optimizer des Students | Adam |
| Loss des Students | Sparse Categorical Crossentropy aus Logits |
| Learning Rate des Students | `1e-4` während Sticky-Rescue |
| Batch Size | `256` während Sticky-Rescue |
| Epochen pro Rescue-Runde | `3` |
| Validation Fraction | `0.05` |
| Shuffle Buffer | `8192` |
| Abgeschlossene Rescue-Runden | `337` |
| Teacher/Student-gelabelte Episoden | `20,202` |
| Teacher/Student-gelabelte Samples | `8,966,962` |
| Bestes 10-Spiele-Ergebnis | `100.0` Durchschnitt |
| Pflicht-5-Spiele-Ergebnis | `99.2` Durchschnitt |

## Environment-Vorverarbeitung

| Parameter | Wert |
| --- | --- |
| Raw Observation | Atari RGB Frame `(210, 160, 3)` |
| Vorverarbeiteter Input | 4 gestapelte Graustufenbilder |
| Modell-Input-Shape | `(4, 84, 84)` |
| Bildgröße | `84` |
| Frame Stack | `4` |
| Frame Skip | `4` |
| No-op Max | `30` |
| Full Action Space | `True` |
| Anzahl Aktionen | `18` |
| Sticky Action Probability | `0.25` |
| Terminal on Life Loss | `False` |

## Neuronales Netz

| Komponente | Wert |
| --- | --- |
| Parameter | `1,093,858` |
| Input dtype | `uint8` |
| Input Scaling | `1 / 255` |
| CNN-Aufbau | IMPALA-ähnliches Residual CNN |
| Conv Sequence Channels | `16`, `32`, `32` |
| Residual Blocks | 2 Residual Blocks pro Conv Sequence |
| Dense Layer | `256` ReLU-Neuronen |
| Output Layer | `18` Logits |

## Baseline DQN / DDQN

Das Projekt enthält zusätzlich den ursprünglichen DQN/DDQN-Code. Dieser war der erste Ansatz vor dem finalen Teacher-Student-Training.

| Parameter | Wert |
| --- | --- |
| Algorithmus | DQN oder DDQN |
| Learning Rate | `1e-4` |
| Discount Factor Gamma | `0.99` |
| Batch Size | `32` |
| Replay Memory Size | `50,000` |
| Replay Type | standardmäßig Prioritized Replay |
| Epsilon Start | `1.0` |
| Epsilon Ende | `0.1` |
| Target Network Update | alle `1000` Trainingsschritte |
| Warmup Steps | `2000` |
| Train Frequency | alle `4` Environment-Schritte |
| Checkpoint Frequency | alle `25` Episoden |
