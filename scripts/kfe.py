import argparse
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from functools import partial
from jax.scipy.stats.norm import cdf, pdf
from matplotlib.animation import FuncAnimation
from mltools.distribution import MixedNormal, Normal
from typing import Callable, Optional, Union

N_ATOMS = 51
LR = 3e-4

KAPPA = 0.1

MIX1 = 0.3

LOC1 = -0.5
LOC2 = 1.
SCALE1 = 0.3
SCALE2 = 0.4

DT = 0.4

REWARD_FUNC = lambda _: 0.
GAMMA = 0.3

DYN_LOC = 1.
DYN_SCALE = 0.3

N_MONTE_CARLO = 1000
N_TRAIN = 100

State = Union[float, jnp.ndarray]

def main(args):
    print(f"Starting {args.n_monte_carlo} Monte Carlo trials...")
    rng = jax.random.PRNGKey(args.seed)
    env = StochasticMunosEnvironment(rng, args)
    env.reset()
    monte_carlo_scores = [env.rollout(args.dt, args.gamma) for _ in range(args.n_monte_carlo)]
    print("Done")

    print("Initializing DEICIDE agent...")
    agent = DeicideAgent(rng, args)
    print("Done")

    fig = plt.figure()
    print("Training begins...")
    ani = FuncAnimation(fig, lambda i: train(agent, env, monte_carlo_scores, i), interval=30)
    plt.show()

def train(agent, env, monte_carlo_scores, i):
    x = env.state
    _, r, done = env.step(agent.dt)
    agent.update(x, r, done)
    bins = agent.n_atoms // 5 + 1
    if done:
        plt.clf()
        plt.hist(monte_carlo_scores,
                 density=True,
                 bins=args.n_monte_carlo // 10,
                 alpha=0.3,
                 label="Monte Carlo")
        plt.hist(agent.particles(agent.params, jnp.array([0.])),
                 density=True,
                 bins=bins,
                 alpha=0.3,
                 label="DEICIDE")
        plt.legend()

def _train(rng, args):
    rngs = hk.PRNGSequence(rng)
    n_atoms = args.n_atoms
    def mlp(x):
        net = hk.Sequential([hk.Linear(16, w_init=hk.initializers.RandomUniform(),
                                           b_init=hk.initializers.TruncatedNormal()),
                             jax.nn.relu,
                             hk.Linear(16, w_init=hk.initializers.RandomUniform(),
                                           b_init=hk.initializers.TruncatedNormal()),
                             jnp.tanh,
                             hk.Linear(n_atoms, w_init=hk.initializers.RandomUniform(),
                                                b_init=hk.initializers.RandomUniform())])
        return net(x)
    func = hk.without_apply_rng(hk.transform(mlp))
    params = func.init(next(rngs), jnp.array([0.]))
    dt = args.dt
    for i in range(args.n_train):
        x = 0.
        while x < 1.:
            r = args.reward_fn(x) * dt
            params = deicide(func, params, args, x, r, False)
            x = x + args.dt * args.dyn_loc + dt * jax.random.normal(next(rngs)) * args.dyn_scale
        r = terminal_score(next(rngs), args)
        params = deicide(func, params, args, x, r, True)
        if check_nan(params):
            raise NumericException(f"[ERROR] Step {i}: params contains nan values")
    return func, params

class DeicideAgent:
    def __init__(self, rng, args):
        self._rngs = hk.PRNGSequence(rng)
        self._n_atoms = args.n_atoms
        # Initialize neural net
        def mlp(x):
            net = hk.Sequential([hk.Linear(16, w_init=hk.initializers.RandomUniform(),
                                            b_init=hk.initializers.TruncatedNormal()),
                                 jax.nn.relu,
                                 hk.Linear(16, w_init=hk.initializers.RandomUniform(),
                                           b_init=hk.initializers.TruncatedNormal()),
                                 jnp.tanh,
                                 hk.Linear(self._n_atoms, w_init=hk.initializers.RandomUniform(),
                                           b_init=hk.initializers.RandomUniform())])
            return net(x)
        self._func = hk.without_apply_rng(hk.transform(mlp))
        self._params = self._func.init(next(self._rngs), jnp.array([0.]))
        self.dt = args.dt
        self.gamma = args.gamma
        self.kappa = args.kappa
        self.lr = args.lr
        self._args = args

        def _compute_target(params, x, r, done):
            particles = self._func.apply(self._params, x)
            grad = jax.jacfwd(partial(self.particles, params))(x).dot(jnp.array([args.dyn_loc]))
            hessian = jax.hessian(partial(self.particles, params))(x).dot(jnp.array([args.dyn_scale])) / 2
            return r + grad.squeeze() + hessian.squeeze() + self.particles(params, x)

        self.compute_target = jax.jit(_compute_target)

    def update(self, x, r, done):
        x = jnp.expand_dims(x.squeeze(), 0)
        target = self.compute_target(self.params, x, r, done)

        target_loss_grads = jax.grad(
            lambda p: (quantile_loss(target, -jnp.log(self.gamma) * self.particles(p, x), self.kappa)
                       / (2 * jnp.log(self.gamma)))
        )(self.params)

        new_params = jax.tree_multimap(lambda x, y: x - self.lr * y, self.params, target_loss_grads)

        trust_grads = jax.grad(
            lambda p: w2_loss(self.particles(self.params, x), self.particles(p, x)) / (2 * self.dt)
        )(new_params)

        self._params = jax.tree_multimap(lambda x, y: x - self.lr * y, new_params, trust_grads)

    @property
    def particles(self):
        return self._func.apply

    @property
    def params(self):
        return self._params

    @property
    def n_atoms(self):
        return self._n_atoms

class StochasticMunosEnvironment:
    def __init__(self, rng, args):
        self.dist = MixedNormal([Normal(args.loc1, args.scale1), Normal(args.loc2, args.scale2)],
                                [args.mix, (1. - args.mix)])
        self.reward_fn = args.reward_func
        self.dyn_dist = Normal(args.dyn_loc, args.dyn_scale)
        self.rngs = hk.PRNGSequence(rng)

    def step(self, dt):
        assert self._state is not None, "Environment hasn't been initialized, call reset()"
        done = False
        self._state += self.dyn_dist.sample(next(self.rngs)) * dt
        if self.state < 1.:
            r = self.reward_fn(self.state)
        else:
            r = self.dist.sample(next(self.rngs))
            done = True
        return self.state, r, done

    def reset(self):
        self._state = 0.
        return self.state

    @property
    def state(self):
        return self._state

    def rollout(self, dt, gamma=1.):
        self.reset()
        score = 0.
        mul = 1.
        done = False
        while not done:
            mul *= gamma ** dt
            _, r, done = self.step(dt)
            score += mul * r
        return score.item()

def deicide(func, params, args, x, r, done):
    X = jnp.array([x])
    target = (1. - done) * target_fn(func, params, X, r, args) + r * done

    gamma = args.gamma

    # Wasserstein Proximal Gradient
    target_loss_grads = jax.grad(
        lambda p: quantile_loss(target, -jnp.log(gamma) * func.apply(p, X), args.kappa) / (2 * jnp.log(gamma))
    )(params)
    new_params = param_update(args.lr, params, target_loss_grads)
    trust_grads = jax.grad(
        lambda p: w2_loss(func.apply(p, X), func.apply(params, X)) / (2 * args.dt)
    )(new_params)
    params = param_update(args.lr, new_params, trust_grads)
    return params

# @jax.jit
def w2_loss(target, pred):
    pred_sorted = pred[jnp.argsort(pred)]
    target_sorted = target[jnp.argsort(target)]
    return jnp.square(pred_sorted - target_sorted).mean()

@jax.jit
def quantile_loss(target, pred, kappa=1e-1):
    def huber(u):
        return jnp.where(jnp.abs(u) <= kappa,
                         jnp.square(u) / (2 * kappa),
                         jnp.abs(u) - kappa / 2)
    n_quantiles = pred.shape[-1]
    midpoints = (jnp.arange(n_quantiles) + 0.5) / n_quantiles
    bellman_diff = target[None, :] - pred[:, None]
    asym_diff = jnp.abs(midpoints[:, None] - (bellman_diff < 0).astype(jnp.float32))
    loss = asym_diff * huber(bellman_diff)
    return jnp.mean(loss, axis=-1).mean()

def check_nan(params):
    return jax.tree_util.tree_reduce(lambda acc, x: (acc or jnp.any(x)),
                                     jax.tree_map(jnp.isnan, params),
                                     False)

class NumericException(Exception):
    def __init__(self, message):
        super(NumericException, self).__init__(message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_atoms", type=int, default=N_ATOMS, help="Number of particles to learn")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--kappa", type=float, default=KAPPA, help="Huber loss threshold")
    parser.add_argument("--mix", type=float, default=MIX1, help="Mixing weight for first Gaussian")
    parser.add_argument("--loc1", type=float, default=LOC1, help="Mean of first Gaussian")
    parser.add_argument("--loc2", type=float, default=LOC2, help="Mean of second Gaussian")
    parser.add_argument("--scale1", type=float, default=SCALE1, help="Scale of first Gaussian")
    parser.add_argument("--scale2", type=float, default=SCALE2, help="Scale of second Gaussian")
    parser.add_argument("--dyn_loc", type=State, default=DYN_LOC, help="Dynamics mean")
    parser.add_argument("--dyn_scale", type=State, default=DYN_SCALE, help="Dynamics scale")
    parser.add_argument("--dt", type=float, default=DT, help="Length of timestep")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Discount factor")
    parser.add_argument("--reward_func", type=Callable[[State], float], default=REWARD_FUNC, help="Reward function")
    parser.add_argument("--n_monte_carlo", type=int, default=N_MONTE_CARLO, help="Number of Monte Carlo trials")
    parser.add_argument("--n_train", type=int, default=N_TRAIN, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
