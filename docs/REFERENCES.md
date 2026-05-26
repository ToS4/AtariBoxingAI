# Quellen

- Gymnasium Boxing Dokumentation: https://gymnasium.farama.org/v0.27.0/environments/atari/boxing/
- CleanRL Boxing Teacher-Modellfamilie: https://huggingface.co/cleanrl/Boxing-v5-cleanba_ppo_envpool_impala_atari_wrapper-seed2
- CleanRL Paper: https://arxiv.org/abs/2111.08819
- EnvPool Paper: https://arxiv.org/abs/2206.10558

Wichtig: Das finale abgegebene Modell ist nicht einfach das externe Teacher-Modell. Das finale Modell ist unser eigenes Keras-Student-Modell `models/mohi_boxer_best.h5`. Es wurde mit Teacher/Student-gelabelten Trajektorien trainiert und danach über Evaluationen als bestes Modell ausgewählt.
