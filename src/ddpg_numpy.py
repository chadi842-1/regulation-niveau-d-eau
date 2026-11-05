"""DDPG en numpy pur (réseaux denses, rétropropagation écrite à la main, Adam).

Choix numpy plutôt que PyTorch : le projet ne dépend que de numpy/gymnasium, et les
poids de l'acteur s'exportent tels quels vers le C embarqué (scripts/export_c.py).

Références : Lillicrap et al. 2016 (DDPG), Silver et al. 2014 (gradient de politique
déterministe). Voir docs/bibliographie.md.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class MLP:
    """Perceptron : couches cachées ReLU, sortie linéaire ou sigmoïde, optimiseur Adam."""

    def __init__(self, sizes, out_sigmoid: bool, rng, lr: float):
        self.sizes, self.out_sigmoid, self.lr = list(sizes), out_sigmoid, lr
        self.W, self.b = [], []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = i == len(sizes) - 2
            lim = 3e-3 if last else np.sqrt(6.0 / n_in)   # init DDPG : dernière couche petite
            self.W.append(rng.uniform(-lim, lim, (n_in, n_out)).astype(np.float32))
            self.b.append(np.zeros(n_out, dtype=np.float32))
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0
        self._cache = None

    def forward(self, x, keep=False):
        a = x
        acts = [x]
        n = len(self.W)
        for i in range(n):
            z = a @ self.W[i] + self.b[i]
            if i < n - 1:
                a = np.maximum(z, 0.0)
            else:
                a = _sigmoid(z) if self.out_sigmoid else z
            acts.append(a)
        if keep:
            self._cache = acts
        return a

    def backward(self, dout, need_params=True):
        """Rétropropage dL/dout. Renvoie (dL/dx, grads) ; utilise le cache du dernier forward(keep=True)."""
        acts = self._cache
        n = len(self.W)
        g = dout
        if self.out_sigmoid:
            y = acts[-1]
            g = g * y * (1.0 - y)
        gW, gb = [None] * n, [None] * n
        for i in reversed(range(n)):
            if i < n - 1:
                g = g * (acts[i + 1] > 0)
            if need_params:
                gW[i] = acts[i].T @ g
                gb[i] = g.sum(axis=0)
            g = g @ self.W[i].T
        return g, (gW, gb)

    def adam_step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        gW, gb = grads
        self.t += 1
        c1, c2 = 1 - beta1**self.t, 1 - beta2**self.t
        for i in range(len(self.W)):
            for p, g, m, v in ((self.W, gW, self.mW, self.vW), (self.b, gb, self.mb, self.vb)):
                m[i] = beta1 * m[i] + (1 - beta1) * g[i]
                v[i] = beta2 * v[i] + (1 - beta2) * g[i] ** 2
                p[i] -= (self.lr * (m[i] / c1) / (np.sqrt(v[i] / c2) + eps)).astype(np.float32)

    def copy_from(self, other, tau=1.0):
        for i in range(len(self.W)):
            self.W[i] = (1 - tau) * self.W[i] + tau * other.W[i]
            self.b[i] = (1 - tau) * self.b[i] + tau * other.b[i]

    def state(self):
        return {f"W{i}": w for i, w in enumerate(self.W)} | {f"b{i}": b for i, b in enumerate(self.b)}

    def load_state(self, d):
        for i in range(len(self.W)):
            self.W[i] = d[f"W{i}"].astype(np.float32)
            self.b[i] = d[f"b{i}"].astype(np.float32)


class ReplayBuffer:
    def __init__(self, obs_dim, capacity=100_000):
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros((capacity, 1), np.float32)
        self.r = np.zeros((capacity, 1), np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.cap, self.n, self.i = capacity, 0, 0

    def add(self, s, a, r, s2):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i] = s, a, r, s2
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.n, batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx]


class DDPG:
    def __init__(self, obs_dim=4, hidden=64, gamma=0.98, tau=0.005,
                 lr_actor=3e-4, lr_critic=1e-3, seed=0):
        self.rng = np.random.default_rng(seed)
        self.gamma, self.tau = gamma, tau
        self.actor = MLP([obs_dim, hidden, hidden, 1], True, self.rng, lr_actor)
        self.critic = MLP([obs_dim + 1, hidden, hidden, 1], False, self.rng, lr_critic)
        self.actor_t = MLP([obs_dim, hidden, hidden, 1], True, self.rng, lr_actor)
        self.critic_t = MLP([obs_dim + 1, hidden, hidden, 1], False, self.rng, lr_critic)
        self.actor_t.copy_from(self.actor)
        self.critic_t.copy_from(self.critic)
        self.obs_dim = obs_dim

    def act(self, obs, noise_std=0.0):
        a = self.actor.forward(np.asarray(obs, np.float32)[None, :])[0, 0]
        if noise_std > 0:
            a += self.rng.normal(0.0, noise_std)
        return float(np.clip(a, 0.0, 1.0))

    def update(self, batch):
        s, a, r, s2 = batch
        B = s.shape[0]

        # --- critique : cible de Bellman y = r + gamma * Q'(s', mu'(s')) ---
        a2 = self.actor_t.forward(s2)
        q2 = self.critic_t.forward(np.concatenate([s2, a2], axis=1))
        y = r + self.gamma * q2
        q = self.critic.forward(np.concatenate([s, a], axis=1), keep=True)
        td = q - y
        _, grads = self.critic.backward(2.0 * td / B)
        self.critic.adam_step(grads)
        critic_loss = float(np.mean(td**2))

        # --- acteur : ascension de Q(s, mu(s)) ---
        mu = self.actor.forward(s, keep=True)
        self.critic.forward(np.concatenate([s, mu], axis=1), keep=True)
        dx, _ = self.critic.backward(-np.ones((B, 1), np.float32) / B, need_params=False)
        _, agrads = self.actor.backward(dx[:, -1:])
        self.actor.adam_step(agrads)

        # --- réseaux cibles (moyenne glissante) ---
        self.actor_t.copy_from(self.actor, self.tau)
        self.critic_t.copy_from(self.critic, self.tau)
        return critic_loss
