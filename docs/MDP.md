# Formalisation en processus de décision markovien (MDP)

Ce document formalise la régulation de niveau comme un MDP, rappelle les résultats utilisés
(opérateur de Bellman, Monte Carlo, martingales) et indique ce qui est **vérifié numériquement**
dans `scripts/bellman_check.py`.

## 1. Système

Bilan de masse de la cuve, avec vidange de Torricelli :

    A · dh/dt = q_max · u − c · √h − d(t)

| Symbole | Sens | Valeur |
|---|---|---|
| h | niveau [m], 0 ≤ h ≤ 2 | état physique |
| u | ouverture de la vanne d'entrée, dans [0, 1] | commande |
| d | soutirage perturbateur [m³/s], échelon à un instant aléatoire | perturbation |
| A, c, q_max | section 0,5 m², vidange 0,01 m^2.5/s, débit max 0,02 m³/s | `TankParams` |

Discrétisation : période d'échantillonnage Δt = 1 s, Euler avec 10 sous-pas.
Point d'équilibre pour une consigne h_ref : u* = (c·√h_ref + d) / q_max.

## 2. Le MDP (S, A, P, r, γ)

- **État** s_t = (e_t, I_t, h_t/H_max, u_{t−1}) avec e = h_ref − h et I = ∫ e dt normalisée et bornée.
  L'intégrale d'erreur est ajoutée à l'état pour que la politique puisse annuler l'erreur statique face à
  une perturbation constante (rôle du terme intégral d'un PI). La consigne est récupérable via h + e.
- **Action** a_t = u_t ∈ [0, 1] (espace continu).
- **Transition** s_{t+1} = f(s_t, a_t, d_t) : dynamique ci-dessus, déterministe sauf pour la perturbation d_t.
- **Récompense** r_t = −|e_{t+1}| − λ·|u_t − u_{t−1}| avec λ = 0,05 : pénalise l'écart à la consigne
  (norme L1, gradient non nul même pour les petites erreurs) et les à-coups de commande.
- **Facteur d'actualisation** γ = 0,98 pour l'apprentissage (horizon effectif ≈ 50 pas).
- **Épisode** de 200 pas, terminé par une limite de temps. Ce n'est pas un état terminal : le buffer de
  rejeu ne marque donc aucune transition comme terminale (bootstrap conservé en fin d'épisode).

Limites de la modélisation : la perturbation d n'est pas observée, le problème est donc en toute rigueur
partiellement observé ; l'intégrale d'erreur joue le rôle de statistique résumée. Le temps restant dans
l'épisode n'est pas dans l'état, la politique apprise est donc stationnaire.

## 3. Opérateur de Bellman et contraction

Pour une politique π, la fonction de valeur V^π(s) = E[Σ_t γ^t r_t | s_0 = s, π] est l'unique point fixe de

    (T^π V)(s) = E_{d}[ r(s, π(s)) + γ · V(s') ]

**Contraction.** Pour deux fonctions bornées V, W :
‖T^π V − T^π W‖_∞ = γ · sup_s |E[V(s') − W(s')]| ≤ γ · ‖V − W‖_∞.
Comme (B(S), ‖·‖_∞) est complet, le théorème du point fixe de Banach donne l'existence et l'unicité de V^π et
la convergence géométrique de V_{k+1} = T^π V_k, de vitesse γ. L'opérateur d'optimalité
(T* V)(s) = max_a E[r + γ V(s')] est également une γ-contraction, ce qui fonde la programmation dynamique et
le Q-learning (Puterman 1994 ; Sutton & Barto 2018).

## 4. Monte Carlo

Le retour actualisé G = Σ_{t<H} γ^t r_t est un estimateur sans biais de V^π(s_0) à la troncature près
(biais ≤ γ^H · r_max/(1−γ)). La moyenne de n trajectoires indépendantes a une variance σ²/n ; le théorème
central limite donne l'intervalle de confiance V̂ ± 1,96·σ/√n utilisé dans le script de vérification.

## 5. Martingales

Soit F_t la filtration engendrée par la trajectoire jusqu'à t et

    M_t = Σ_{i<t} γ^i r_i + γ^t · V^π(s_t).

D'après l'équation de Bellman, E[M_{t+1} | F_t] = Σ_{i<t} γ^i r_i + γ^t (r_t + γ E[V^π(s_{t+1}) | s_t]) = M_t.
Le processus (M_t) est donc une **martingale** et E[M_t] = V^π(s_0) pour tout t.

Intérêt pour la convergence des algorithmes : une itération de TD ou de Q-learning s'écrit
Q_{k+1} = Q_k + α_k (T Q_k − Q_k + ε_k), où le bruit d'échantillonnage ε_k est une **différence de
martingale**. Avec les pas de Robbins-Monro (Σ α_k = ∞, Σ α_k² < ∞), la contraction de T et le théorème de
convergence des presque-surmartingales de Robbins-Siegmund, on obtient la convergence presque sûre dans le
cas tabulaire (Jaakkola, Jordan & Singh 1994 ; Tsitsiklis 1994).

**Ce que cela ne garantit pas.** DDPG utilise des réseaux de neurones et des cibles lentes : les hypothèses
du cas tabulaire ne sont plus satisfaites, et aucune garantie de convergence n'est établie ici. L'instabilité
observée en fin d'entraînement (coût de validation qui remonte vers l'épisode 560) est cohérente avec cette limite.

## 6. DDPG

- Critique Q_φ(s, a) : minimise (Q_φ(s,a) − y)², avec y = r + γ Q_φ'(s', μ_θ'(s')).
- Acteur μ_θ(s) ∈ [0, 1] (sigmoïde) : ascension du gradient déterministe de politique (Silver et al. 2014)
  ∇_θ J = E[ ∇_a Q_φ(s, a)|_{a=μ_θ(s)} · ∇_θ μ_θ(s) ].
- Réseaux cibles mis à jour par moyenne glissante, τ = 0,005 ; rejeu d'expérience ; bruit gaussien
  d'exploration décroissant de 0,2 à 0,03 ; Adam. Implémentation numpy dans `src/ddpg_numpy.py`,
  rétropropagation vérifiée par différences finies (`tests/test_ddpg.py`).

## 7. Vérifications numériques (`scripts/bellman_check.py`)

Cadre réduit pour que l'espérance soit calculable : état h seul, consigne fixe 1 m, politique fixe
proportionnelle saturée, soutirage d ~ U[0 ; 0,002] à chaque pas, récompense −|h_ref − h|, γ = 0,98.

1. **Contraction.** Sur 1139 itérations de V_{k+1} = T^π V_k, le rapport ‖ΔV_{k+1}‖ / ‖ΔV_k‖ reste ≤ γ = 0,98
   (maximum observé 0,9800).
2. **Monte Carlo = point fixe.** 4000 trajectoires par état initial :

   | h_0 | Monte Carlo (IC 95 %) | Point fixe de Bellman |
   |---|---|---|
   | 0,3 | −9,2855 ± 0,0022 | −9,2886 |
   | 0,7 | −2,6438 ± 0,0011 | −2,6454 |
   | 1,0 | −0,4335 ± 0,0008 | −0,4338 |
   | 1,4 | −3,4883 ± 0,0012 | −3,4904 |

   Les écarts (≤ 3·10⁻³) sont de l'ordre de l'erreur d'interpolation de la grille (401 points) ; ils dépassent
   l'intervalle de confiance Monte Carlo seul, ce que le script tolère via `3·IC + 0,05`.
3. **Martingale.** Depuis h_0 = 0,3, E[M_t] reste égal à V(s_0) = −9,2886 à l'intervalle de confiance près
   pour t ∈ {1, 5, 20, 60, 150} (valeurs entre −9,2888 et −9,2869).
