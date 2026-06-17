# Maze world-model — experiments log

Journal des runs et de l'investigation du pipeline maze (branche `maze`).
Dates en 2026. Cluster DALIA (GB200), checkpoints sous
`$EBJEPA_CKPTS/ac_video_jepa/maze/` (i.e. `$WORK/checkpoints/...`).

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

## 5. Expériences capacité-vs-algo (lancées 2026-06-02) — RÉSULTATS

| Job | Config | Résultat eval |
|-----|--------|---------------|
| 66216 → 66217 | small 11×11, modèle 32 | eval **crashait** (mismatch géométrie, voir §7) ; re-évalué 68617 → **0 %** |
| 66218 → 66219 | big 21×21, modèle 128 | **0 %**, `mean_state_dist` 54.2 (euclidien) |

**Verdict : même le petit labyrinthe (11×11) avec petit modèle échoue (0 %).** Ça
pointe vers l'**algorithme de planning** plus que vers la capacité du modèle :
agrandir (128) ou rétrécir le maze ne suffit pas. Re-eval 2026-06-08 avec la
métrique A\* (§6) : small `mean_state_dist` géodésique **72 px** (l'agent part
à 105, descend à 72, puis se bloque → progrès partiel, jamais le but).

---

## 6. Eval : distance géodésique A\* (2026-06-08)

`MazeEnv.eval_state` utilisait la distance **euclidienne** `‖goal − curr‖`,
trompeuse en labyrinthe : deux points peuvent être proches à vol d'oiseau mais
séparés par des murs → la métrique créditait le planner d'être « proche » sans
route courte. Remplacée par la **distance géodésique A\*** : positions →
cellules → `solve_a_star` → `(len(path)-1) × cell_size`. Succès = agent sur la
case but. Fallback euclidien si une position arrondit sur un mur. Commit dans
`eb_jepa/datasets/maze/env.py`. Validée en live (la distance suit l'agent, pas
de crash) sur les checkpoints existants.

Fix lié (`main.py`) : l'eval construit la config d'env en mergeant `cfg.data`
(géométrie du modèle) avec la section `data` du yaml d'eval — sinon une
géométrie non-défaut (11×11) n'atteignait pas l'env → encodeur mal dimensionné
(crash `mat1/mat2`). On **retire le bloc `pipeline`** de ce merge (l'env d'eval
ne stream pas ; sinon `device must be provided when pipeline.mode='stream'`).

---

## 7. Exp C — entraînement long-horizon (2026-06-09) — RÉSULTAT NÉGATIF

**Insight de départ.** Le planner déroule le world-model sur `plan_length=90`, mais le
modèle n'était entraîné qu'à `nsteps=8`. Au-delà de ~8 pas ses prédictions
divergent → MPPI optimise contre un modèle faux → l'agent se bloque. Hypothèse :
entraîner long (nsteps=32) + planifier à horizon aligné (plan_length=32) débloque.

**Résultat (train 68624 + eval 68627) — le 0 % PERSISTE.**

| Métrique | Valeur | Lecture |
|---|---|---|
| `pred` final (train) | **0.0142** | dynamique très bien apprise |
| `probe` final (train) | **0.6729** | latente encode bien la position (vs **4–11** pour dim32) |
| eval `success_rate` | **0.0 %** | jamais le but |
| eval `mean_state_dist` (A\*) | **235.5 px** | l'agent bouge, reste loin / coincé |

→ Conséquence forte : **l'hypothèse capacité/représentation est réfutée par les
chiffres**. `probe=0.67` prouve que le modèle 128 *sait* où est l'agent, et
`pred=0.014` qu'il *sait* prédire la dynamique. Le long-horizon n'a pas aidé non
plus. **Il ne reste qu'un coupable : l'objectif de planning greedy** (`repr_dist`,
distance-au-but dans la latente) — il pousse l'agent droit vers le but → dans les
murs → minimum local. C'est le pattern « bouge, se rapproche, se bloque » observé
en light-eval (`state_dist` coincé ~198). Élimination cumulée : génération (§1),
compute (§2), capacité (§5), horizon d'entraînement (§7). **Reste l'algo de planning.**

**Micro-bench CUDA (`scripts/maze_bench.py`, jobs 68615/68620)** — coût/step réel
(fwd+bwd compilé, B=384, GB200) :

| Config | ms/step |
|---|---|
| REF d32 T17 n8 (actuel) | 99.6 |
| d128 T17 n8 (modèle seul) | 123.0 |
| d128 T49 **n8** (frames seules) | 320.5 |
| **PROP d128 T49 n32** (Exp C) | **358.8** |

→ Les **`nsteps` sont quasi gratuits** (8→32 : +38 ms, +12 %) ; le coût vient du
`sample_length` (frames encodées). La crainte « 32 pas latents = lent » est
infondée. Exp C ≈ 359 ms/step (~1.7 min/epoch hors light-evals).

**Config `train_maze_long.yaml` (`exp_long_h32`, job 68624)** : 21×21, modèle
128, `nsteps=32`, `sample_length=49`, `min_path_length=50`, GPU-stream + 32
workers A\*, compile `reduce-overhead`, bf16. Apprend bien (`pred` 1.6 → 0.30).

**Eval finale `afterok` (job 68627)** : `eval_maze_long.sbatch`, métrique A\*,
`planning_mppi_h32.yaml` (`plan_length=32` **aligné** sur l'horizon entraîné).
→ 0 % (table ci-dessus). La tension horizon-vs-chemin devient secondaire : même
avec un modèle long-horizon fiable, l'objectif greedy ne sort pas du labyrinthe.

---

## 8. Prochaines étapes — viser l'objectif de planning

Tout pointe maintenant vers l'**objectif de planning greedy**, pas le world-model
(génération, compute, capacité, représentation, horizon : tous éliminés). Pistes,
par ordre de préférence :

1. **Sous-buts / waypoints.** Calculer le chemin A\* sur le wall-mask, en extraire
   des points intermédiaires (~tous les 8–16 px), et faire planifier MPPI vers le
   **prochain waypoint** plutôt que le but final. Transforme un problème global non
   convexe en une suite de cibles localement atteignables (où `repr_dist` greedy
   marche déjà — c'est ce qui rendait two_rooms faisable). Cible principale.
2. **Objectif de planning non-greedy** : pénaliser la collision/mur dans le coût
   MPPI, ou une distance apprise (value/cost-to-go) au lieu de `repr_dist`.
3. Vérifier que MPPI a assez de samples/horizon pour *trouver* le détour
   (augmenter `num_samples`, élargir le bruit d'action) — moins probable vu §4.

---

## 9. Pistes de planning (2026-06-12) — sous-buts + collision : NÉGATIF

Les deux pistes de §8 sont **côté planning uniquement** : le world-model
`exp_long_h32` (probe=0.67, pred=0.014) est inchangé, on **réévalue** le
checkpoint avec une logique de planning différente → runs courtes (~5–8 min,
eval-only, 12 épisodes, `n_allowed_steps=140`, `stop_on_success`). Implémentation :
`MazeEnv.compute_waypoints` (A\* → sous-buts) + logique waypoint dans `main_eval` ;
`ReprDistCollisionMPCObjective` (latent décodé via probe → cellule → pénalité mur).

| Job | Piste | Config | `success` | `mean_dist` (A\*) |
|-----|-------|--------|-----------|-------------------|
| 70124 | #1 Waypoints | `spacing=4`, `plan_length=12` | **0.0 %** | 260.75 |
| 70125 | #2 Collision | `coeff=0.3`, `plan_length=32` | **0.0 %** | 254.25 |
| (réf §7) | greedy nu | `plan_length=32` | 0.0 % | 235.5 |

**Les deux échouent, et finissent même PLUS LOIN que le greedy nu** (l'agent
erre / part dans le mauvais sens). Pattern « bouge à peine / s'éloigne » inchangé.

**Lecture #1 :** `spacing=4` laisse encore des **murs entre l'agent et le
sous-but** → le greedy `repr_dist` retombe dans son minimum local à petite
échelle. Le régime vraiment local (sans mur intermédiaire) n'est atteint qu'à
`spacing=1` (sous-but = cellule adjacente).

**Sweep `spacing` (jobs 70126/70127)** : `spacing=1` **0%**, `spacing=2` **0%**.
Même avec un sous-but *adjacent* (zéro mur possible entre l'agent et la cible),
l'agent échoue et **ne bouge quasi pas**. → Le blocage n'est **PAS** l'objectif
global de planning. **La conclusion §7/§8 (« reste l'objectif greedy ») était
fausse** — mauvais coupable.

---

## 10. Cause racine RÉELLE + premier succès (2026-06-12)

Après avoir éliminé l'objectif global (sweep §9), diagnostic instrumenté
(`_diagnose_world_model`, job 70142) : à l'épisode 0, pour chaque action
cardinale on compare le déplacement **prédit par le world-model** (décodé via le
probe) à ce que ferait l'env. Plus deux fixes successifs :

### 10.1 Fix exécution — snap des actions (NÉCESSAIRE mais pas suffisant)
Les actions d'entraînement maze sont **cardinales × cell_size** (`dirs*3`,
axe-alignées) et `action_encoder = nn.Identity()` → le predictor n'a vu que
`{(±3,0),(0,±3)}`. MPPI échantillonnait des gaussiennes continues `N(0,2)` sur
les deux axes → **hors-distribution** → prédictions incohérentes. Fix
`snap_actions_to_grid` (`GCAgent._snap_to_grid`) : snap des actions sur la grille
cardinale × cell_size avant le world-model (= ce que fait `env.step`). Seul, le
snap ne suffit pas (jobs 70136/70137 toujours 0%).

### 10.2 Cause racine — `repr_dist` noyé par le masque de murs
Le diagnostic montre que **l'encodage d'action est correct** (right → +colonne,
down → +ligne) et le **probe est bon** (`4.48,4.90` vs vrai `4,4`). Le modèle
prédit bien les latents (`pred=0.014`). Mais l'obs maze est **2 canaux
(dot + masque de murs)** : le masque statique **domine le latent**, la position
du dot y pèse peu → `repr_dist(pred, but)` est **quasi-plat** → MPPI sans signal
→ agent figé. (two_rooms marche car structure plus simple, la position pèse plus.)

**Fix — planifier en espace de POSITION via le probe** (`ProbePositionMPCObjective`,
`objective_type: probe_pos`) : objectif = distance entre la position prédite
(décodée par le probe) et la cible — signal positionnel direct, invariant aux murs.

### 10.3 Premier succès
Recette = **snap + `probe_pos` + waypoints A\*** (eval-only, world-model Exp C
inchangé, ~4 min/run) :

| Job | Config | `success` | `mean_dist` |
|-----|--------|-----------|-------------|
| 70146 | snap+probe_pos+**waypoints sp=2** | **25 %** | 122.75 |
| 70147 | snap+probe_pos+**greedy** (no wp) | 0 % | 252.25 |

→ **Premier succès non-nul (25%)**, `state_dist` descend nettement (261→75).
Le greedy seul (sans waypoints) erre encore → **les waypoints restent
nécessaires** (greedy global colle aux murs même avec bon signal local).
Les trois ingrédients sont requis. Tuning du `spacing` en cours (jobs 70149-70151).

## 11. Coût de planning APPRIS (value TD-MPC) vs distance géométrique

Remplacement du coût « distance-dans-le-latent » par une **value apprise**
`V(z, z_goal)` (TD-MPC ; Hansen et al.) entraînée par **TD(0) sur les rollouts du
world-model lui-même** (target net EMA, reward 1 à l'arrivée, régression sur
latents réels + imaginés). Voir `examples/ac_video_jepa/maze/README_value.md`.

- **`GoalValueHead`** (`state_decoder.py`), **`LearnedValueMPCObjective`**
  (`planning.py`, `objective_type: learned_value`, coût = `1 − V`), entraînement
  gated par `value_coeff` + `freeze_world_model` (value-only sur le world-model gelé).
- Protocole : world-model **gelé** (= `exp_aux_pos`), value head entraînée seule
  (loss 0.035→0.010, 6 ep). Comparaison contrôlée, mêmes 16 mazes, seul le coût change.

| Régime | learned VALUE | probe_pos | repr_dist |
|--------|--------------:|----------:|----------:|
| greedy (but global) | 0 % | 0 % | 0 % |
| A\* waypoints sp=1 | **12.5 %** | 6.25 % | — |
| A\* waypoints sp=2 (tuné) | **37.5 %** | 6.25 % | — |

→ **Avec waypoints, la value apprise fait ~6× le coût-distance** (37.5 vs 6.25 %),
même world-model / mêmes mazes / mêmes réglages MPPI. Validation de l'hypothèse :
un coût-à-apprendre corrélé au succès guide bien mieux que la distance brute.
Greedy-global reste 0 % (but au-delà de l'horizon entraîné de la value). Limiteur
restant = l'horizon de la value (fenêtres 16 pas) ; plot `results/maze_value/`.
