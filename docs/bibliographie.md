# Étude bibliographique : DDPG vs PID pour la commande en boucle fermée

Références citées de mémoire (auteurs, titre, support, année) ; vérifier les détails de pagination avant toute
citation académique formelle.

## Commande classique (PID)

- Åström, K. J., Hägglund, T. *PID Controllers: Theory, Design, and Tuning*, 2e éd., ISA, 1995.
  Référence sur le PID discret, l'anti-windup et le réglage. Sert de base de comparaison ici.

## Cadre MDP et programmation dynamique

- Bellman, R. *Dynamic Programming*, Princeton University Press, 1957. Principe d'optimalité.
- Puterman, M. L. *Markov Decision Processes: Discrete Stochastic Dynamic Programming*, Wiley, 1994.
  Contraction de l'opérateur de Bellman, existence et unicité des points fixes.
- Sutton, R. S., Barto, A. G. *Reinforcement Learning: An Introduction*, 2e éd., MIT Press, 2018.
  Introduction générale : Monte Carlo, TD, Q-learning, méthodes de gradient de politique.

## Convergence (approximation stochastique et martingales)

- Robbins, H., Monro, S. « A stochastic approximation method », *Annals of Mathematical Statistics*, 22(3), 1951.
  Conditions sur les pas Σ α_k = ∞, Σ α_k² < ∞.
- Robbins, H., Siegmund, D. « A convergence theorem for non negative almost supermartingales and some
  applications », in *Optimizing Methods in Statistics*, Academic Press, 1971.
- Watkins, C. J. C. H., Dayan, P. « Q-learning », *Machine Learning*, 8, 1992.
- Jaakkola, T., Jordan, M. I., Singh, S. P. « On the convergence of stochastic iterative dynamic programming
  algorithms », *Neural Computation*, 6(6), 1994.
- Tsitsiklis, J. N. « Asynchronous stochastic approximation and Q-learning », *Machine Learning*, 16, 1994.
  Ces trois derniers établissent la convergence presque sûre du Q-learning tabulaire.

## Apprentissage par renforcement à action continue

- Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., Riedmiller, M. « Deterministic policy gradient
  algorithms », *ICML*, 2014. Théorème du gradient de politique déterministe.
- Lillicrap, T. P. et al. « Continuous control with deep reinforcement learning », *ICLR*, 2016 (arXiv:1509.02971).
  Algorithme DDPG : acteur-critique, rejeu d'expérience, réseaux cibles.
- Fujimoto, S., van Hoof, H., Meger, D. « Addressing function approximation error in actor-critic methods »,
  *ICML*, 2018 (TD3). Biais de surestimation du critique ; pistes de stabilisation de DDPG.
- Kingma, D. P., Ba, J. « Adam: a method for stochastic optimization », *ICLR*, 2015. Optimiseur utilisé.

## Synthèse comparative DDPG vs PID

| Critère | PID | DDPG |
|---|---|---|
| Modèle de la cuve nécessaire | Non, mais réglage des gains | Non pour l'agent, mais un simulateur est nécessaire pour s'entraîner |
| Réglage | 3 gains, quelques essais | Réseaux, récompense, bruit, nombreux hyperparamètres |
| Garanties | Analyse fréquentielle classique | Pas de garantie de convergence avec réseaux de neurones |
| Fonction de coût | Implicite dans les gains | Explicite : n'importe quelle récompense, y compris non différentiable |
| Non-linéarités, contraintes | Saturation gérée par anti-windup | Apprises, saturation intégrée à la sortie du réseau |
| Coût d'exécution | Quelques opérations par pas | 4545 paramètres, ~4,4 k MAC par pas |

Résultats de ce dépôt sur ce critère : voir `README.md`.
