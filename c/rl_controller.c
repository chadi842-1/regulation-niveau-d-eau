#include "rl_controller.h"

#include <math.h>

static float clampf(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

/* sortie[j] = relu(b[j] + somme_i entree[i] * W[i * n_out + j]) ; W stocké ligne par ligne */
static void dense_relu(const float *in, int n_in, const float *W, const float *b,
                       float *out, int n_out)
{
    for (int j = 0; j < n_out; j++) {
        float acc = b[j];
        for (int i = 0; i < n_in; i++) {
            acc += in[i] * W[i * n_out + j];
        }
        out[j] = acc > 0.0f ? acc : 0.0f;
    }
}

float rl_policy(const float obs[RL_OBS_DIM])
{
    float h1[RL_H1];
    float h2[RL_H2];

    dense_relu(obs, RL_OBS_DIM, RL_W0, RL_B0, h1, RL_H1);
    dense_relu(h1, RL_H1, RL_W1, RL_B1, h2, RL_H2);

    float z = RL_B2[0];
    for (int i = 0; i < RL_H2; i++) {
        z += h2[i] * RL_W2[i];
    }
    return 1.0f / (1.0f + expf(-z));
}

void rl_init(rl_state_t *s, float u0)
{
    s->integ = 0.0f;
    s->u_prev = clampf(u0, 0.0f, 1.0f);
    s->started = 0;
}

float rl_step(rl_state_t *s, float h, float h_ref)
{
    const float e = h_ref - h;

    /* Convention de l'environnement Python : l'intégrale vaut 0 à l'instant initial,
     * puis intègre l'erreur de chaque nouvel échantillon. */
    if (s->started) {
        s->integ += e * RL_DT;
    }
    s->started = 1;

    float obs[RL_OBS_DIM];
    obs[0] = e;
    obs[1] = clampf(s->integ / RL_INT_SCALE, -2.0f, 2.0f);
    obs[2] = h / RL_H_MAX;
    obs[3] = s->u_prev;

    const float u = clampf(rl_policy(obs), 0.0f, 1.0f);
    s->u_prev = u;
    return u;
}
