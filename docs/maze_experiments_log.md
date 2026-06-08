# Maze world-model — experiments log

Journal des runs et de l'investigation du pipeline maze (branche `maze`).
Dates en 2026. Cluster DALIA (GB200), checkpoints sous
`/lustre/work/pdl17890/udl806719/checkpoints/ac_video_jepa/maze/`.

---

## 1. Perf de génération de données — RÉSOLU

`GPUMazeGenerator._sample_batch_cpu` faisait du DFS + A* en boucle Python
séquentielle (GIL-bound), seul goulot du pipeline GPU stream.

| | Avant | Après |
|---|---|---|
| Échantillonnage | 1 thread, ~9.4 ms/sample | `ProcessPoolExecutor` 32 workers (spawn) |
| Stall de `swap()` par chunk (3840) | ~34 s | **~0.3–0.4 s** |
| Épisode d'entraînement | epoch ~16 min | epoch ~5–6 min |

Wiring : `utils._init_gpu_stream` lit `pipeline.num_gen_workers` →
`init_gpu_maze_data` → `GPUMazePipelineManager` → `GPUMazeGenerator`.
`num_workers<=1` garde le chemin séquentiel. Commit `1df9eba`.

---

## 2. Perf du pas d'entraînement — RÉSOLU (le goulot était la génération)

Le run de réf 66188 tournait à **~1.20 s/it** vs **0.136 s/it** pour two_rooms.
On a longtemps soupçonné le modèle 32-dim (les runs post-fix 66216/66218 à
~0.15 s/it semblaient indiquer que le modèle 128 était « 8× plus rapide » sur la
même entrée). **Faux.** 66188 datait d'**AVANT** le fix de génération parallèle
(`1df9eba`) ; 66216/66218 sont **après**. La différence était la **génération
A\* séquentielle**, pas le modèle.

### Preuve : micro-benchmark CUDA-events (job 68615, `scripts/maze_bench.py`)

Step réel (fwd+bwd compilé `reduce-overhead`, bf16, B=384) sur **données
synthétiques** (isole le compute du modèle, sans génération) :

| Condition | ms/step | fwd / bwd |
|---|---|---|
| **A** dim32, img63 (le « lent ») | **100.1** | 42.7 / 57.4 |
| B dim128, img63 | 122.4 | 45.5 / 76.6 |
| C dim32, img33 | 94.7 | 42.0 / 52.4 |
| D dim32, img63 + `cudnn.benchmark` | 99.7 | 42.2 / 57.3 |

**Verdict :**
- Le compute du modèle « lent » (dim32, img63) = **100 ms**, identique à tout le
  reste. Le modèle 128 est même **légèrement plus lent** (122 ms), comme attendu.
  → l'hypothèse « 32-dim pathologiquement lent / kernel cuDNN dégénéré » est
  **réfutée**.
- `cudnn.benchmark` ne change rien (99.7 vs 100.1) → pas de mauvais algo de conv.
- Les runs post-fix à ~0.15 s/it = 100 ms compute + ~50 ms data/probe. Cohérent.

**Il ne reste aucun goulot perf** : le maze tourne au régime attendu (~0.15 s/it)
depuis le fix de génération `1df9eba`. (Aux écartées par mesure/lecture déjà :
`compile_mode`, loader `__iter__`, nb de frames, `gen_batch_size`, partage GPU.)

---

## 3. Run de référence — job 66188 (21×21, modèle 32-dim)

- 12 epochs, 1 h 11, ~330 s/epoch (budget 1 h 30, fini).
- **`pred` final = 0.0020** — excellent (< two_rooms 0.018) → dynamique locale bien apprise.
- **`probe` explosé : 0.9 → 4–11** (vs ~0.19 pour two_rooms).

> Le `probe` est **détaché** de l'encodeur (`probe_optimizer` séparé,
> ne met à jour que `xy_head`). Il ne corrompt donc PAS la représentation —
> c'est un diagnostic. Sa valeur élevée **révèle** que la latente 32-dim
> n'encode pas linéairement la position de l'agent dans le labyrinthe.

### Eval (job 66190) — ÉCHEC
- **0 % de succès** (0/5 épisodes avant annulation), ~19.7 min/épisode.
- Lenteur expliquée : l'agent n'atteint jamais le but → chaque épisode épuise
  les `n_allowed_steps=200` (pas de terminaison anticipée comme two_rooms ~30 steps).

---

## 4. Cause racine du 0 % — deux problèmes distincts

1. **Capacité de représentation.** `henc=hpre=dstc=32`. Suffisant pour two_rooms
   (1 mur + 1 porte), probablement trop petit pour encoder la topologie d'un
   labyrinthe 21×21 + la position. Cohérent avec le `probe` explosé.
2. **Objectif de planning.** `planning_mppi.yaml` minimise `repr_dist`
   (distance au but dans l'espace de représentation). En labyrinthe, ça pousse
   l'agent **droit vers le but → dans les murs** (minima locaux). Limitation
   algorithmique classique de la planification par distance-au-but avec obstacles.

Notes :
- MPPI ne clippe **pas** `max_norms` (`planning.py:374`) → l'échelle d'action
  n'est pas le blocage.
- `MazeEnv` : but = coin opposé `(H-2, W-2)`, succès = case but (seuil 1 case).
- Actions maze = ±`cell_size` (3 px) par pas cardinal, **non normalisées**
  (`MazeNormalizer` n'a pas de `normalize_action`).

---

## 5. Expériences en cours (lancées 2026-06-02)

Objectif : **localiser** (capacité vs algo) en parallèle.

| Job | Config | Idée |
|-----|--------|------|
| 66216 train → 66217 eval | `train_maze_small.yaml` — **11×11** (img 33), modèle **32** | Si un petit labyrinthe se résout, le 0 % vient de la capacité/échelle, pas de l'algo |
| 66218 train → 66219 eval | `train_maze_big.yaml` — 21×21, modèle **128** | Si `probe` chute et l'eval réussit → la capacité était le facteur limitant |

- Evals déclenchées par `--dependency=afterok:<train>`.
- `EXP_FOLDER` **fixe** (sans timestamp) pour coupler train/eval :
  `.../maze/exp_small_sanity/impala_seed1` et `.../maze/exp_big128/impala_seed1`.
- Configs : `examples/ac_video_jepa/cfgs/{train_maze_small,train_maze_big,eval_maze_small,eval_maze_bounded}.yaml`.
- sbatch : `scripts/{train,eval}_maze_{small,big}.sbatch`.

**Signal le moins cher** : le `probe` loss pendant le training du modèle 128-dim.
S'il chute vers ~0.2 → capacité. S'il reste haut → algo de planning.

---

## 6. Prochaines étapes

1. Vérifier le démarrage de 66216/66218 (pas de crash sur géométrie 11×11 / modèle 128).
2. Comparer `probe` + succès eval des deux expériences → conclure capacité vs algo.
3. (séparé) Micro-benchmark du 1.20 s/it si on veut la vraie cause perf.
4. Selon le verdict : agrandir encore le modèle, ou changer l'objectif de planning
   (sous-buts, replanning court, objectif non-greedy).
