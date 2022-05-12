import argparse
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.signal as signal

from functools import partial
from mltools.distribution import Normal, MixedNormal, parse_distribution
from tqdm import tqdm
from typing import Any, Callable, List, Optional, Tuple, Union

State = float
Reward = float
DiscretizedState = int

Atom = float
EmpiricalDistribution = List[Atom]

def main(args):
    plt.style.use(args.style)

    rngs = hk.PRNGSequence(args.seed)

    dist = parse_distribution(args.dist_type, args.dist_params)
    dynamics = Normal(args.dyn_loc, args.dyn_scale)

    reward_fn = jax.jit(
        lambda rng, x: (x >= 1.) * dist.sample(rng)
    )

    dm = DiscreteMapper(args)
    print(f"Discretized to {dm.n_states} states")
    print(f"Reward distribution at endpoint: N({dist.loc}, {dist.scale})")
    data = [[] for _ in range(dm.n_states + 1)]

    for _ in tqdm(range(args.n)):
        x = 0.
        returns = []
        states = []
        score = 0.
        while True:
            x = max(0., min(x + dynamics.sample(next(rngs)) * args.tau, 1.))
            states.append(x)
            score += reward_fn(next(rngs), x)
            returns.append(score)
            if x >= 1:
                break

        states = jax.vmap(dm.discrete_state)(jnp.array(states).squeeze())
        returns = signal.lfilter([1], [1, -args.gamma ** args.tau], returns[::-1])[::-1]
        for (i, r) in zip(states, returns):
            data[i].append(r)

    plot_monte_carlo(data, dm)
    quantile_data = get_quantiles(data, dm.n_atoms)
    plot_quantiles(quantile_data, dm)
    errors = quantile_errors(quantile_data,
                             dm,
                             args.dyn_loc,
                             args.dyn_scale,
                             lambda x: 2. * (x >= 1.), args.gamma)
    fig, ax = plt.subplots()
    im = ax.imshow(errors[2:-2,3:-3].clip(0,4.))
    ax.figure.colorbar(im, ax=ax)
    plt.show()

def quantile_errors(data, dm, dyn_loc, dyn_scale, reward_fn, gamma, loss='l2'):
    delta = dm.delta
    xs = dm.state_representative(jnp.arange(dm.n_states + 1))
    lgy = jnp.log(gamma)
    grad_x = finite_difference_tabular(data, 0, dm.delta)
    grad_q = finite_difference_tabular(data, 1, dm.zeta)
    grad_term_x = (dyn_loc * grad_x).squeeze()
    assert grad_term_x.shape == (dm.n_states + 1, dm.n_atoms), \
        f"Grad_x has incorrect shape: {grad_term_x.shape}"

    """
    (r + q * log(gamma)) * dQdq
    --------v-----------   --v--
    (NS,) + (NS, NQ)       (NS,NQ) ==> (NS, NQ)
    """
    grad_term_q = ((reward_fn(xs)[:, None] + lgy * data) * grad_q).squeeze()
    assert grad_term_q.shape == (dm.n_states + 1, dm.n_atoms), \
        f"Grad_q has incorrect shape: {grad_term_q.shape}"

    hessian_x = finite_difference_tabular(grad_x, 0, dm.delta)
    hessian_term = 0.5 * (dyn_scale ** 2) * hessian_x.squeeze()
    assert hessian_term.shape == (dm.n_states + 1, dm.n_atoms), \
        f"Hessian has incorrect shape: {hessian_term.shape}"

    residual = grad_term_x + grad_term_q + hessian_term

    if loss == 'l2':
        return 0.5 * jnp.square(residual)
    if loss == 'l1':
        return jnp.abs(residual)
    raise NotImplementedError(f"Unsupported loss function: \"{loss}\"")

def finite_difference_tabular(data, axis, delta):
    return (jnp.roll(data, -1, axis=axis) - jnp.roll(data, 1, axis=axis)) / (2 * delta)

def get_quantiles(data, n_quantiles):
    quantiles = (jnp.arange(n_quantiles) + 0.5) / n_quantiles
    data = list(map(jnp.array, data))
    return jnp.array([jax.vmap(partial(jnp.quantile, d))(quantiles) for d in data])

def plot_quantiles(data, dm, num_obs=8, height=5, save=False):
    fig, axs = plt.subplots(1, num_obs, figsize=(height * num_obs/2, height), dpi=80)
    midpoints = (jnp.arange(dm.n_atoms) + 0.5) / dm.n_atoms
    xs = jnp.linspace(0., 1., num_obs)
    xs_discrete = jax.vmap(dm.discrete_state)(xs)
    width = 1. / dm.n_atoms
    for (i, ax) in zip(xs_discrete, axs):
        ax.set_xlim(0, 1)
        ax.bar(midpoints, data[i], width=width, edgecolor='black')
        ax.set_title(f"x = {xs[i]:>.3f}")
    plt.show()

def plot_monte_carlo(data, dm, num_obs=8, height=5, save=False):
    fig, axs = plt.subplots(1, num_obs, figsize=(height * num_obs/2, height), dpi=80)
    xs = jnp.linspace(0., 1., num_obs)
    xs_discrete = jax.vmap(dm.discrete_state)(xs)
    # fig.tight_layout()
    for (i, ax) in zip(xs_discrete, axs):
        ax.hist(data[i], density=True, bins=50)
        vals = jnp.array(data[i])
        mean = jnp.mean(vals)
        _, M = ax.get_ylim()
        ax.set_ylim(ymin=0, ymax=M)
        ax.plot([mean, mean], [0, M], '--',
                color='yellow', alpha=0.5, label=f"mu: {mean:.3f}")
        ax.set_title(f"x = {dm.state_representative(i):>.3f}")
        ax.legend()

    if save:
        plt.savefig(fig, "monte_carlo_plot.png")
    else:
        plt.show()

class DiscreteMapper:
    def __init__(self, args):
        self.delta = args.delta
        self.n_atoms = args.n_atoms
        self._n_states = int(1. / self.delta)
        self._zeta = 1. / self.n_atoms

    def discrete_state(self, x: State) -> DiscretizedState:
        return (x / self.delta).astype(jnp.int32)

    def state_representative(self, x: DiscretizedState) -> State:
        return jnp.array(x).astype(jnp.float32) * self.delta

    @property
    def n_states(self):
        return self._n_states

    @property
    def zeta(self):
        return self._zeta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-n", type=int, default=1000, help="Number of Monte Carlo trials")
    parser.add_argument("--delta", type=float, default=1e-1, help="Space discretization")
    parser.add_argument("--n_atoms", type=int, default=51)
    parser.add_argument("--tau", type=float, default=1e-1, help="Timestep")
    parser.add_argument("--gamma", type=float, default=0.3, help="Discount factor")
    parser.add_argument("--dist_type", type=str, default="normal")
    parser.add_argument("--dist_params", type=str, default="2 1")
    parser.add_argument("--dyn_loc", type=float, default=1.)
    parser.add_argument("--dyn_scale", type=float, default=0.5)
    parser.add_argument("--style", type=str, default="seaborn")
    args = parser.parse_args()
    main(args)
