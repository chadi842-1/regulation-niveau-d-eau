/* Banc de test du contrôleur C (hors cible, sur PC).
 *
 *   test_policy vectors <fichier.csv>
 *       relit des lignes "obs0,obs1,obs2,obs3,attendu" produites par Python et affiche l'écart max.
 *   test_policy sim <h0> <h_ref> <t_dist> <d_mag> <u0>
 *       simule la cuve (même modèle que src/tank_env.py) en boucle fermée avec le contrôleur C
 *       et affiche le niveau à chaque pas.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rl_controller.h"

static double tank_step(double h, double u, double d)
{
    if (u < 0.0) u = 0.0;
    if (u > 1.0) u = 1.0;
    const double sub_dt = PLANT_DT / PLANT_N_SUB;
    for (int k = 0; k < PLANT_N_SUB; k++) {
        const double dh = (PLANT_Q_MAX * u - PLANT_C_OUT * sqrt(h > 0.0 ? h : 0.0) - d) / PLANT_AREA;
        h += dh * sub_dt;
        if (h < 0.0) h = 0.0;
        if (h > PLANT_H_MAX) h = PLANT_H_MAX;
    }
    return h;
}

static int run_vectors(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        perror("fopen");
        return 2;
    }
    char line[256];
    double max_err = 0.0;
    int n = 0;
    while (fgets(line, sizeof line, f)) {
        float obs[RL_OBS_DIM], expected;
        if (sscanf(line, "%f,%f,%f,%f,%f", &obs[0], &obs[1], &obs[2], &obs[3], &expected) != 5) continue;
        const double err = fabs((double)rl_policy(obs) - (double)expected);
        if (err > max_err) max_err = err;
        n++;
    }
    fclose(f);
    printf("vectors n=%d max_abs_err=%.3e\n", n, max_err);
    return n > 0 ? 0 : 3;
}

static int run_sim(double h0, double h_ref, int t_dist, double d_mag, double u0)
{
    rl_state_t st;
    rl_init(&st, (float)u0);
    double h = h0;
    for (int t = 0; t < PLANT_HORIZON; t++) {
        const float u = rl_step(&st, (float)h, (float)h_ref);
        h = tank_step(h, (double)u, t >= t_dist ? d_mag : 0.0);
        printf("%.9f\n", h);
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 3 && strcmp(argv[1], "vectors") == 0) return run_vectors(argv[2]);
    if (argc == 7 && strcmp(argv[1], "sim") == 0)
        return run_sim(atof(argv[2]), atof(argv[3]), atoi(argv[4]), atof(argv[5]), atof(argv[6]));
    fprintf(stderr, "usage: %s vectors <csv> | sim <h0> <h_ref> <t_dist> <d_mag> <u0>\n", argv[0]);
    return 1;
}
