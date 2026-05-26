# Abgabe-Checkliste

Diese Tabelle zeigt, wo die Anforderungen aus `docs/ABGABE.pdf` in diesem Ordner erfüllt sind.

| Anforderung | Datei / Befehl |
| --- | --- |
| Lauffähiges Python-Projekt | `src/`, `requirements.txt`, `README.md` |
| README mit Abhängigkeiten und Startbefehl | `README.md` |
| Trainierter Agent | `models/mohi_boxer_best.h5` und `models/mohi_boxer_best.keras` |
| Agent direkt ladbar / spielbar | `python src/boxing.py play --algorithm dqn --model models/mohi_boxer_best.h5 --episodes 1` |
| Trainingskurve | `artifacts/training/training_reward_curve.png` |
| Daten zur Trainingskurve | `artifacts/training/training_reward_curve.csv` |
| Evaluation gegen Standard-Atari-Gegner | `artifacts/evaluation/final_5_game_summary.json` |
| Durchschnittsreward über 5 Spiele | `99.2` |
| Evaluationsvideos | `artifacts/evaluation/videos/boxing-eval-episode-*.mp4` |
| Präsentation | `presentation/MohiBoxer_presentation.pptx` und `presentation/MohiBoxer_presentation.md` |
| Schriftliche Dokumentation | `docs/PROJECT_DOCUMENTATION.md` |
| Hyperparameter-Tabelle | `docs/HYPERPARAMETERS.md` |
| Schriftliche Reflexion | `docs/REFLECTION.md` |
| Zusammengefasstes Dokument als PDF | `docs/MohiBoxer_Dokumentation.pdf` |

Empfohlenes Modell für die Abgabe: `models/mohi_boxer_best.h5`.
