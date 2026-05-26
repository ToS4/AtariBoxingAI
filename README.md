# MohiBoxer - RL Boxing Agent

MohiBoxer ist ein Keras-Agent für das Atari-Spiel `ALE/Boxing-v5`.

Finales Abgabe-Modell:

- `models/mohi_boxer_best.h5`
- `models/mohi_boxer_best.keras`

Wenn nur eine Datei abgegeben werden soll, dann `models/mohi_boxer_best.h5` verwenden.

## Ergebnis

Pflicht-Evaluation über 5 Spiele gegen den eingebauten Atari-Gegner:

- Rewards: `98, 100, 100, 100, 98`
- Durchschnittsreward: `99.2`
- Videos: `artifacts/evaluation/videos/`

Bestes 10-Spiele-Ergebnis während des Trainings:

- Rewards: zehnmal `100`
- Durchschnittsreward: `100.0`

## Installation

Empfohlen: Python `3.11` oder `3.12`. TensorFlow in diesem Projekt ist nicht für Python `3.14` gedacht.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
AutoROM --accept-license
```

Linux / WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
AutoROM --accept-license
```

Optionale Turnier-Pakete:

```bash
python -m pip install "pettingzoo[atari]" supersuit
```

## Agent Starten

Finales Modell über 5 Spiele evaluieren:

```bash
python src/boxing.py evaluate --algorithm dqn --model models/mohi_boxer_best.h5 --episodes 5 --video --output-dir artifacts/evaluation_check
```

Ein Spiel sichtbar abspielen:

```bash
python src/boxing.py play --algorithm dqn --model models/mohi_boxer_best.h5 --episodes 1
```

Baseline-DDQN von neu trainieren:

```bash
python src/boxing.py train --output-dir artifacts/boxing_ddqn --episodes 300 --algorithm ddqn
```

Teacher-Student-Modell aus vorhandenen Trajektorien trainieren:

```bash
python src/distill_boxing.py train --dataset-dir artifacts/teacher_boxing_dataset_sticky_rescue --output-dir artifacts/student_imitation --epochs 5
```

## GPU-Speicher

Standardmäßig läuft `src/boxing.py` auf CPU und setzt `CUDA_VISIBLE_DEVICES=-1`.

GPU-Modus:

```bash
BOXING_TF_DEVICE=gpu python src/boxing.py evaluate --algorithm dqn --model models/mohi_boxer_best.h5 --episodes 5
```

Im GPU-Modus nutzt der Code:

```python
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)
```

Dadurch reserviert TensorFlow nicht direkt den gesamten GPU-Speicher.

## Ordnerstruktur

- `src/` - Python-Code für Training, Evaluation, Distillation und Turnier-Helfer
- `models/` - finales `.h5` und `.keras` Modell
- `artifacts/evaluation/` - 5-Spiele-Evaluation und Videos
- `artifacts/training/` - Reward-Kurve, Trainingsübersicht und Modellzusammenfassung
- `docs/` - schriftliche Dokumentation, Hyperparameter, Reflexion und Checkliste
- `presentation/` - Präsentation und Sprecherhilfe

## Wichtige Dokumente

- `docs/PROJECT_DOCUMENTATION.md`
- `docs/HYPERPARAMETERS.md`
- `docs/REFLECTION.md`
- `docs/EVALUATION_REPORT.md`
- `docs/SUBMISSION_CHECKLIST.md`
- `docs/MohiBoxer_Dokumentation.pdf`

Gymnasium Boxing Referenz: https://gymnasium.farama.org/v0.27.0/environments/atari/boxing/
