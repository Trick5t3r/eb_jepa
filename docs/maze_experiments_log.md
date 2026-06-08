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

## 2. Perf du pas d'entraînement — NON RÉSOLU

Maze tourne à **~1.20 s/it** vs **0.136 s/it** pour two_rooms, à **modèle et
forme d'entrée identiques** (`(384, 2, 17, ~63²)`, impala, mêmes hyperparamètres).

Hypothèses **écartées par mesure / lecture** :

| Cause suspectée | Verdict |
|---|---|
| Génération de données (swap) | ❌ +0.28 s seulement, tous les 10 steps |
| `compile_mode` (CUDA graphs) | ❌ `reduce-overhead` ajouté (commit `692aaca`) → **aucun gain** |
| Loader `__iter__` | ❌ code partagé, même `index_select` |
| Nb de frames rendues | ❌ two_rooms rend aussi la trajectoire complète |
| `gen_batch_size` | ❌ `None` des deux côtés |
| Partage GPU (co-tenants) | ❌ GPU dédié (4 GB200 / nœud) |

Le fait que `reduce-overhead` n'apporte rien prouve que le step **n'est pas
launch-bound**. **Conclusion : non tranchable en lisant le code** — nécessite un
micro-benchmark CUDA-events (data-gen vs forward vs backward).

### Nouvel indice (runs 66216 / 66218, 2026-06-02 ~23h25)

Surprise : les deux nouveaux runs tournent à **~0.15 s/it**, PAS 1.2 s/it.

| Run | Géométrie | Modèle | Vitesse | Epoch 0 |
|-----|-----------|--------|---------|---------|
| 66188 (réf) | 21×21 (img 63) | **32** | ~1.2 s/it | ~330 s |
| 66218 | 21×21 (img 63) | **128** | ~0.158 s/it (6.31 it/s) | 41 s |
| 66216 | 11×11 (img 33) | 32 | ~0.150 s/it (6.66 it/s) | 39 s |

66218 a la **même entrée** que 66188 mais un modèle **plus gros**, et il est
**~8× plus rapide**. Donc le modèle plus large est plus rapide que le petit sur
la même entrée → forte piste : le modèle minuscule 32-dim ne capturait pas
correctement en CUDA graph / était dominé par l'overhead (kernels trop petits
pour saturer le GB200), contrairement au 128. À confirmer par micro-benchmark.

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
