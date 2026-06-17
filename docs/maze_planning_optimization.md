# Maze planning — campagne d'optimisation du win-rate (2026-06-12)

Journal vivant de la campagne pour faire **réussir l'agent à atteindre la sortie**
du labyrinthe 21×21, puis maximiser le taux de succès. Tout est **eval-only** sur
le checkpoint déjà entraîné `exp_long_h32/impala_seed1` (world-model Exp C,
`pred=0.014`, `probe=0.67`) — **aucun réentraînement** sauf mention contraire.
Voir aussi `maze_experiments_log.md` §9–§10 (cause racine).

## TL;DR
- **RÉSULTAT FINAL : 0% → 93.75%** (15/16) de succès, l'agent atteint la sortie.
  Config gagnante = `planning_mppi_probe_escape_p1.yaml` :
  **snap + probe_pos + waypoints sp=1 + action_prior A\* + stall-escape A\*
  (déclencheur = non-progrès géodésique, `stall_patience=1`)**. Eval-only sur
  exp_long_h32 (zéro training retenu). Hybride : MPC world-model + fallback A\*
  (le MPC fait ~moitié des coups, A\* corrige les cellules-pièges).
- **Point de départ : 0% de succès**, attribué à tort (§7/§8) à « l'objectif de
  planning greedy ».
- **Vraie cause** (en 2 couches) :
  1. **Actions hors-distribution** : le world-model n'a vu que des actions
     cardinales × cell_size ; MPPI lui envoyait des gaussiennes continues → fix
     **`snap_actions_to_grid`**.
  2. **`repr_dist` noyé par le masque de murs** : l'obs maze = 2 canaux
     (dot + murs statiques) ; le latent est dominé par les murs → objectif plat.
     Fix : **planifier en espace de position via le probe** (`probe_pos`).
- **Recette gagnante : `snap` + `probe_pos` + `waypoints A*`** → **premier succès
  25%** (job 70146). Les 3 ingrédients sont nécessaires (greedy seul = 0%).
- En cours : tuning `spacing` / `plan_length` / `n_allowed_steps` pour monter le taux.

## Implémentation (worktree `eb_jepa_maze`, branche `maze`)
- `eb_jepa/datasets/maze/env.py` : `MazeEnv.compute_waypoints(spacing)` — sous-buts
  le long du chemin A\*.
- `eb_jepa/planning.py` :
  - `GCAgent._snap_to_grid` + flag `planner.snap_actions_to_grid` — snap des
    actions sur la grille cardinale × cell_size avant le world-model.
  - `ProbePositionMPCObjective` (`objective_type: probe_pos`) — coût = distance
    position-décodée-par-probe ↔ cible (waypoint).
  - `ReprDistCollisionMPCObjective` (`repr_dist_collision`) — repr_dist + pénalité mur (abandonné, 0%).
  - waypoints + `stop_on_success` dans `main_eval` ; `_diagnose_world_model` (diag 1-pas).
- Configs : `examples/ac_video_jepa/maze/cfgs/planning_mppi_*.yaml`,
  `eval_maze_short.yaml` (12 ép./140), `eval_maze_med.yaml` (16 ép./180).
- sbatch générique : `scripts/eval_maze_planning.sbatch <plan_cfg> <suffix> [eval_cfg]`.

## Toutes les runs

| Job | Recette | success | mean_dist | Note |
|-----|---------|---------|-----------|------|
| 68627 | (réf §7) repr_dist greedy plan=32 | 0% | 235.5 | point de départ |
| 70124 | waypoints sp=4 + repr_dist | 0% | 260.75 | erre |
| 70125 | collision coeff=0.3 + repr_dist | 0% | 254.25 | erre |
| 70126 | waypoints sp=1 + repr_dist | 0% | 242.25 | figé → pas l'objectif global |
| 70127 | waypoints sp=2 + repr_dist | 0% | 246.0 | figé |
| 70136 | snap + greedy repr_dist | 0% | 241.75 | snap seul insuffisant |
| 70137 | snap + waypoints sp=2 + repr_dist | 0% | 250.75 | bouge à peine |
| 70142 | **DIAG** 1-pas | — | — | action OK, probe OK, repr_dist plat |
| 70146 | **snap + probe_pos + waypoints sp=2** | **25%** | **122.75** | 🎉 1er succès (12 ép.) |
| 70147 | snap + probe_pos + greedy (no wp) | 0% | 252.25 | greedy seul erre → wp nécessaires |
| 70151 | snap + probe_pos + wp sp=2, plan=8 | 12.5% | 142.7 | round1 (16 ép.) — variance vs 25% à 12 ép. |
| 70149 | snap + probe_pos + wp sp=1, plan=6 | ANNULÉ | | trop lent (ép. ratés = 180 pas pleins) |
| 70150 | snap + probe_pos + wp sp=3, plan=12 | ANNULÉ | | idem |
| 70168 | snap + probe_pos + wp sp=1, **plan=2** | 0% | 181.7 | plan trop court |
| 70169 | snap + probe_pos + wp sp=1, **plan=4** | 16.7% | 121.7 | round2 |
| 70170 | snap + probe_pos + wp sp=2, **plan=4** | 16.7% | **95.0** | round2 — meilleur mean_dist |
| 70173 | + prior A\* sp=1, plan=4 | 16.7% | 142.2 | round3 |
| 70174 | + prior A\* **sp=2, plan=4** | 16.7% | **80.5** | round3 — meilleur mean_dist |
| 70215 | **fine-tune aux-position** (6 ép.) | — | — | pred=0.001 mais **reg=0.07 effondré** |
| 70216 | eval modèle fine-tuné (prior sp2) | **6.25%** | 245.6 | ❌ collapse → pire |
| 70222 | + stall-escape v1 (vise waypoint) | 12.5% | 167 | ❌ bug : boucle dans un mur hors-path |
| 70223 | + stall-escape v2 (re-A\* depuis cellule) | 31.2% | 89.8 | mieux, mais MPC oscille en arrière |
| 70225 | + escape v3 (**détecteur géodésique**) | **62.5%** | 30.0 | 🎉 capture stalls+oscillations |
| 70258 | + escape v3 **`stall_patience=1`** | **93.75%** | **3.94** | 🏆 config finale (16 ép.) |
| 70xxx | confirmation 30 ép. (200 pas) | _en cours_ | | estime fiable |

### ⚠️ Décision utilisateur : PAS de fallback — world-model SEUL
Le 93.75% est un hybride (A\* fait ~moitié des coups). L'utilisateur le rejette :
le world-model doit naviguer **seul** (A\* toléré seulement comme sous-buts
high-level, mais aucun coup exécuté par A\*). → cible = maximiser le world-model
MPC **sans fallback** (base ~25-31%) via un **retrain** (autorisé, ≤45 min).

### Round 7 — retrain aux-position CORRIGÉ (anti-collapse)
1ᵉʳ retrain (round 4) collapse car `aux_pos_coeff=2.0` → loss aux dominait
(5.4/5.4) → VICReg écrasé. Correctif : `aux_pos_coeff=0.5`, `lr=2e-4` (au lieu
de 5e-4), 7 épochs, surveillance de `reg` (kill si effondrement). Eval `afterok`
**sans fallback** (`planning_mppi_probe_prior_wp2_pl4.yaml` : snap+probe_pos+
waypoints+prior, le world-model choisit chaque coup). Jobs 70286/70287.

### Round 8 — la VRAIE cause racine du plafond world-model : pas de collisions
Le retrain aux-position (round 7) n'a rien donné (25%, pas de collapse) → la
décodabilité du probe n'était pas le limiteur. Vrai diagnostic (rappel du job
70142) : action "down" vers un **mur** → le modèle prédit **+1.7px de mouvement**
(devrait rester sur place). **Le world-model ne sait pas prédire les collisions.**
Cause : les trajectoires d'entraînement sont des **chemins A\*** → **jamais un
coup dans un mur** → le modèle apprend « action X → bouge dans la direction X »
universellement, sans collisions. Au planning, MPPI propose un coup vers un mur,
le modèle prédit (à tort) un progrès → l'agent fonce/oscille → stalls.

**Fix (données) : `data.wall_bump_prob`** — injecte dans les trajectoires des
"wall bumps" (action cardinale vers un mur, position inchangée) pour apprendre
les collisions (`generate_path_and_actions`, validé numpy : bumps tapent bien
des murs, but atteint). Fine-tune depuis exp_long_h32, `wall_bump_prob=0.3`,
`aux_pos_coeff=0` (isolé), nsteps=8, 7 épochs. Eval **sans fallback**
(`probe_prior_wp2`). Jobs 70330/70331. Hypothèse : le modèle évite enfin les
murs au planning → le world-model navigue seul nettement mieux.

**RÉSULTAT round 8 (jobs 70330/70331) : world-model SEUL = 56.25%** (dist 13.5),
vs 25% avant. `reg=1.28` (sain, pas de collapse), `pred=0.0068`. **La cause
racine était bien l'absence de collisions dans les données** — pas la
décodabilité du probe ni l'objectif. Plus de 2× le plafond, SANS fallback.
→ on pousse : confirmation 30 ép., variantes planning (sp1, sans prior),
2ᵉ train `wall_bump_prob=0.5`.

**Sweep planning sur le modèle collision (jobs 70364-66, world-model seul) :**
| config | success | dist | n_ép |
|---|---|---|---|
| sp2 + prior | 46.7% | 77.7 | 30 (fiable) |
| sp1 + prior | 31.2% | 72 | 16 |
| **sp2 SANS prior** | **50%** | 29.4 | 16 |

→ **Le prior A\* ne sert quasi plus** (sp2 sans prior ≈ avec) : A\* ne fait QUE
poser des sous-buts, le world-model MPC trouve la direction seul (MPPI from
scratch). Config world-model-seul retenue = **snap + probe_pos + waypoints sp2,
SANS prior, SANS fallback** ≈ **47-56%**. sp1 moins bon. Bilan : 0% → 25% (planning)
→ **~50% (collision-aware world-model, navigation autonome)**.

**Sweet spot `wall_bump_prob` :** `0.3` ≈ 47-56%, mais `0.5` (job 70378) **= 6.25%**
(`reg=2.04` sain mais le modèle voit 50% de "stays" → sur-apprend à rester sur
place → aucune action ne semble progresser au planning → bloqué). Trop de
collisions tue la navigation. **Optimum ≈ 0.3.** Confirmation finale 30 ép. du
modèle `bump=0.3` (sp2 sans prior) en cours.

### 🏆 Round 6 — `stall_patience=1` = 93.75% (hybride, rejeté par l'utilisateur)
Avec patience=3 le MPC gaspillait 3 pas à osciller avant chaque correction A\* →
échecs step-limités. **`stall_patience=1`** : A\* corrige dès le 1er pas sans
progrès géodésique, le MPC ne garde la main que sur les pas où il avance vraiment.
→ **93.75% (15/16), mean_dist 3.94**. Le seul échec (dist=18) est step-limité sur
un maze très long → confirmation à 200 pas. Fallback A\* : 1–45 coups/épisode
(le MPC fait l'autre moitié). C'est l'objectif atteint : l'agent atteint la sortie.

### Round 5 — stall-escape : itérations
Architecture = **MPC world-model + fallback A\* sur non-progrès** (fraction
fallback loggée/épisode). 3 itérations clés :
1. **v1** (vise le waypoint courant) : 12.5% — bug, si l'agent dévie hors-path le
   waypoint n'est plus adjacent → fallback pointe dans un mur → boucle infinie.
2. **v2** (`env.astar_action_from_current` : re-A\* depuis la cellule courante,
   robuste partout) : 31.2% — mais le MPC reprend la main entre deux fallbacks et
   fait **reculer** l'agent (oscillation), non détectée par le critère mouvement.
3. **v3** (déclencheur = **distance géodésique qui ne décroît plus**, capture
   stalls ET oscillations) : **62.5% / dist 30**. Échecs restants = step-limités
   (le MPC gaspille `stall_patience` pas à osciller). → test `stall_patience=1`.

### Round 4 — VERDICT : fine-tune aux-position ÉCHOUE (collapse)
Le fine-tune à LR frais (5e-4, optimizer reset) a **cassé l'équilibre VICReg** :
`reg` 0.07 (effondré), `pred=0.001` trivialement satisfait → **collapse de
représentation** → latent ne distingue plus les positions → eval **6.25% /
dist 245** (pire que 25%). Risque JEPA classique du fine-tuning agressif.
**Abandonné** — le 25% eval-only reste la référence. (Infra : 2 OOM avant ça car
SLURM plaçait le job sur dalianvl03 dont le GPU0 est squatté → `--exclude=dalianvl03`.)

### Round 5 — stall-escape (fallback A\* sur cellule-piège, eval-only)
L'agent échoue **uniquement** par stalls. Fix `stall_escape` : si l'agent n'a pas
bougé depuis `stall_patience` pas, exécuter directement le coup A\* vers le
sous-but adjacent (spacing=1 → garanti ouvert), 1 pas, puis rendre la main au MPC.
Architecture = **MPC world-model + fallback A\* sur stall** (fraction de fallback
loggée par épisode). Sur le bon modèle exp_long_h32.

### Mode d'échec (trajectoires de 70174)
L'agent **descend puis STALLE** à une cellule-piège (valeurs plates : `225 225
225`, `30 30 30`) — ce n'est PAS un manque de pas. Un épisode n'a jamais bougé
(`225×6`). Même avec le prior A\*, MPPI mésprédit l'action à ces cellules.
→ **Limiteur = représentation/world-model** (latent pas assez position-net),
pas le planner. Plafond eval-only ~17-25% (mean_dist 80-95).

### Round 4 — fine-tune avec loss auxiliaire de position (training ≤45 min)
Fix principiel : le probe est **détaché** → n'influence pas l'encodeur, donc le
latent n'encode la position que faiblement (`probe=0.67`) → `probe_pos` bruité →
stalls. Ajout `model.aux_pos_coeff` (`main.py`) : terme `mse(head(encode(x)), loc)`
**non-détaché** ajouté à la loss JEPA → modèle l'encodeur pour un latent
position-décodable. Fine-tune court depuis exp_long_h32 (poids only, optimizer
frais via `meta.init_from`), `sample_length=17 / nsteps=8` (~5-6 min/ép.),
6 épochs, `aux_pos_coeff=2`, `min_path_length=50` (eval même difficulté).
→ sauve dans `exp_aux_pos/`, eval `afterok` avec snap+probe_pos+prior+wp sp=2.

### Round 3 — prior d'action A\* (anti-bruit-du-probe)
Le win-rate plafonne ~17-25% ; `mean_dist` se stabilise ~95 → l'agent fait
~60% du chemin puis se bloque. Limiteur identifié : **bruit de décodage du
probe** (~2 cellules sur les latents prédits) vs cellule = 3 px → à certaines
jonctions MPPI choisit le mauvais cardinal → oscillation/déviation. Fix
**`waypoint_action_prior`** : warm-start de la moyenne MPPI sur le cardinal A\*
vers le sous-but courant (`MPPIPlanner.action_prior`), le world-model + `probe_pos`
ne font plus que confirmer/corriger → robuste au bruit du probe.

### Optimisation runtime
- Flag **`meta.skip_unroll_eval`** ajouté (`main.py`) : saute la rollout-eval
  (GIFs de qualité de prédiction, inutile au win-rate) → runs de tuning plus
  rapides. Ajouté au sbatch générique.
- Round 1 trop lent car les épisodes ratés tournent les 180 pas pleins ×
  16 ép. × 3 jobs/nœud. Round 2 = eval court (12 ép./140) + `plan_length`
  court matché au spacing + skip_unroll.

## Notes méthodo
- **Variance du win-rate** : à 12–16 épisodes l'estimation est très bruitée
  (même réglage sp=2 : 25% sur 12 ép. vs 12.5% sur 16 ép.). `mean_state_dist`
  est plus stable pour comparer. Pour le réglage final → relancer avec ≥30 ép.
- Runtime : ~4 min/run isolé ; ~10–20 min quand 3 jobs partagent un nœud.

## Prochaines étapes
1. Finir le sweep `spacing` (1/2/3) → choisir le meilleur sur `mean_dist`.
2. 2ᵉ ordre autour du gagnant : `plan_length` court (3–6, concentre le signal
   probe sur l'action immédiate), `num_act_stepped`, `n_allowed_steps`.
3. Confirmer le meilleur réglage sur ≥30 épisodes (estime fiable).
4. Si plafond ~30–40% → retrain court ≤45 min (modèle nsteps=8, ~6 min/epoch)
   pour une dynamique/probe plus nets.
