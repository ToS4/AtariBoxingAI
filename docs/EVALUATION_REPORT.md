# Evaluationsbericht

## Umgebung

- Environment: `ALE/Boxing-v5`
- Gegner: eingebauter Atari-Boxing-Gegner
- Observation: 4 gestapelte Graustufenbilder mit `84x84` Pixeln
- Action Space: `Discrete(18)`
- Sticky Actions: `repeat_action_probability=0.25`

## Pflicht-Evaluation Über 5 Spiele

Lokaler Befehl zur erneuten Evaluation:

```bash
python src/boxing.py evaluate --algorithm dqn --model models/mohi_boxer_best.h5 --episodes 5 --video --output-dir artifacts/evaluation_check
```

Abgabe-Modell:

```text
models/mohi_boxer_best.h5
```

Ergebnis:

| Spiel | Reward |
| --- | ---: |
| 1 | 98 |
| 2 | 100 |
| 3 | 100 |
| 4 | 100 |
| 5 | 98 |

Durchschnittsreward:

```text
99.2
```

Standardabweichung:

```text
0.9798
```

Lokale Beweise:

- Zusammenfassung: `artifacts/evaluation/final_5_game_summary.json`
- Videos: `artifacts/evaluation/videos/`

## Beste 10-Spiele-Evaluation

Während des Trainings wurde ein perfektes 10-Spiele-Ergebnis erreicht:

- 10 Spiele
- alle 10 Rewards waren `100`
- Durchschnitt: `100.0`
- Standardabweichung: `0.0`

Lokale Beweise:

- `artifacts/training/best_10_game_evaluation.json`
- `artifacts/training/training_overview.json`
