import numpy as np

from src.ddpg_numpy import DDPG, MLP, ReplayBuffer


def _loss(net, x, w):
    return float(np.sum(net.forward(x) * w))


def test_backward_matches_finite_differences():
    rng = np.random.default_rng(0)
    for out_sigmoid in (False, True):
        net = MLP([4, 8, 8, 1], out_sigmoid, rng, lr=1e-3)
        # grossit la dernière couche pour que le test ne soit pas trivial
        net.W[-1] = rng.normal(0, 0.5, net.W[-1].shape).astype(np.float64)
        net.W = [w.astype(np.float64) for w in net.W]
        net.b = [b.astype(np.float64) for b in net.b]
        x = rng.normal(size=(5, 4))
        w = rng.normal(size=(5, 1))
        net.forward(x, keep=True)
        dx, (gW, gb) = net.backward(w)

        eps = 1e-6
        for i in range(len(net.W)):
            for idx in [(0, 0), (1, 0)] if net.W[i].shape[1] == 1 else [(0, 0), (1, 2)]:
                old = net.W[i][idx]
                net.W[i][idx] = old + eps
                lp = _loss(net, x, w)
                net.W[i][idx] = old - eps
                lm = _loss(net, x, w)
                net.W[i][idx] = old
                assert abs((lp - lm) / (2 * eps) - gW[i][idx]) < 1e-5
        # gradient par rapport à l'entrée
        xp, xm = x.copy(), x.copy()
        xp[2, 1] += eps
        xm[2, 1] -= eps
        num = (_loss(net, xp, w) - _loss(net, xm, w)) / (2 * eps)
        assert abs(num - dx[2, 1]) < 1e-5


def test_critic_learns_a_constant_target():
    agent = DDPG(seed=1)
    rng = np.random.default_rng(1)
    s = rng.normal(size=(64, 4)).astype(np.float32)
    a = rng.uniform(size=(64, 1)).astype(np.float32)
    first = last = None
    for k in range(300):
        loss = agent.update((s, a, np.full((64, 1), -1.0, np.float32), s))
        first = loss if first is None else first
        last = loss
    assert last < first


def test_replay_buffer_wraps():
    buf = ReplayBuffer(4, capacity=3)
    for k in range(5):
        buf.add(np.full(4, k), 0.5, -1.0, np.full(4, k + 1))
    assert buf.n == 3
    assert set(buf.s[:, 0]) == {2.0, 3.0, 4.0}
