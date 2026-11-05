/* Contrôleur de niveau par politique DDPG, pour cible embarquée temps réel.
 *
 * - aucune allocation dynamique, tailles fixes connues à la compilation
 * - un pas de commande = ~4,4 k multiplications-accumulations (réseau 4-64-64-1)
 * - seule dépendance libm : expf() (sigmoïde de sortie)
 *
 * Les poids viennent de policy_weights.h, généré par scripts/export_c.py.
 * Les observations sont construites exactement comme dans src/tank_env.py.
 */
#ifndef RL_CONTROLLER_H
#define RL_CONTROLLER_H

#include "policy_weights.h"

typedef struct {
    float integ;     /* intégrale de l'erreur [m.s] */
    float u_prev;    /* dernière commande appliquée, dans [0, 1] */
    int   started;   /* 0 tant que le premier échantillon n'a pas été traité */
} rl_state_t;

/* u0 : commande initiale supposée (même convention que l'environnement Python) */
void  rl_init(rl_state_t *s, float u0);

/* Politique pure a = mu(obs), obs[4] = {e, I_norm, h/H_MAX, u_prev}. */
float rl_policy(const float obs[RL_OBS_DIM]);

/* À appeler à chaque période RL_DT avec la mesure de niveau h et la consigne.
 * Retourne l'ouverture de vanne dans [0, 1]. */
float rl_step(rl_state_t *s, float h, float h_ref);

#endif
