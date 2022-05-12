import argparse
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from functools import partial
from jax.scipy.stats.norm import pdf, cdf
from matplotlib.animation import FuncAnimation
from mltools.distribution.normal import Normal, MixedNormal
from typing import Any, Callable, List, Optional, Union

# EXPERIMENT-SPECIFIC IMPORTS
from kfe import w2_loss
from kfe import quantile_loss

def main(args):
    init_style_stuff(args)
    dist = parse_distribution(args.dist_type, args.dist_params)
    loss_fn = parse_loss(args.loss)
    key = jax.random.PRNGKey(args.seed)
    rngs = hk.PRNGSequence(key)

    print("Begins Monte Carlo sampling...")
    subs = jax.random.split(next(rngs), args.n_monte_carlo)
    monte_carlo_samples = jax.vmap(dist.sample)(subs).squeeze()
    print("Done")

    fig = plt.figure()
    particles = ParticleCluster(next(rngs), args.n_particles, args.lr, args.kappa)

    ani = FuncAnimation(fig,
                        partial(particlestep,
                                args,
                                loss_fn,
                                sort_particles=True,
                                rngs=rngs,
                                target_dist=dist,
                                particles=particles,
                                dist=dist,
                                monte_carlo_samples=monte_carlo_samples),
                        interval=args.i)

    plt.show()

def particlestep(args,
                 loss_fn,
                 i,
                 rngs=None,
                 target_dist=None,
                 particles=None,
                 monte_carlo_samples=None,
                 sort_particles=False,
                 dist=None,
                 **kwargs):
    plt.clf()
    target = target_dist.sample(next(rngs))
    if args.debug:
        particles.monitor(target)
    target = target * jnp.ones_like(particles.get.squeeze())
    if args.debug:
        pred = particles.get.squeeze()
        loss_fn(target, pred)
    loss, grad = jax.value_and_grad(lambda pred: loss_fn(target, pred.squeeze()))(particles.get)
    particles.update(grad)

    monte_carlo_bins = int(0.05 * args.n_monte_carlo)
    particle_bins = min(monte_carlo_bins, args.n_particles)
    sample_bins = min(len(particles.empirical), monte_carlo_bins)

    xmin = min(jnp.min(monte_carlo_samples), jnp.min(particles.get))
    xmax = max(jnp.max(monte_carlo_samples), jnp.max(particles.get))
    xs = jnp.linspace(xmin, xmax, 500)

    plt.hist(monte_carlo_samples,
             density=True,
             bins=monte_carlo_bins,
             label="Monte Carlo")
    plt.hist(particles.get.squeeze(),
             density=True,
             bins=particle_bins,
             alpha=0.5,
             label="Particles")
    if args.debug:
        plt.hist(particles.empirical,
                 density=True,
                 bins=particle_bins,
                 alpha=0.5,
                 label="Training samples")
    if dist is not None:
        plt.plot(xs, dist.pdf(xs), '--', alpha=0.6, label="True density")
    plt.xlim((xmin, xmax))
    plt.legend()

def parse_distribution(dist_type, dist_params):
    dist_type = dist_type.lower()
    if dist_type == "normal":
        return Normal.instantiate(dist_params)
    if dist_type == "mixed_normal":
        return MixedNormal.instantiate(dist_params)
    raise NotImplementedError(f"Distribution type \"{dist_type}\" not recognized")

def parse_loss(loss: str) -> Callable[[Any], float]:
    loss = loss.lower()
    if (loss == "w2") or (loss == "wasserstein2"):
        return w2_loss
    if (loss == "w1") or (loss == "wasserstein1"):
        return partial(quantile_loss, kappa=0.)
    if loss == "quantile_huber":
        return quantile_loss
    raise NotImplementedError(f"Loss function \"{loss}\" not recognized")

def init_style_stuff(args):
    plt.style.use(args.style)

class ParticleCluster(object):
    def __init__(self, rng, n_particles, lr=1e-2, kappa=1e-1):
        self._particles = jnp.sort(10. * jax.random.uniform(rng, shape=(n_particles,)) - 5.)
        self._n_particles = n_particles
        self._lr = lr
        self._kappa = kappa
        self._empirical = []

    def update(self, grads):
        self._particles = self._particles - self._lr * grads

    def monitor(self, sample):
        self._empirical.append(sample.item())

    @property
    def get(self):
        return self._particles

    @property
    def n(self):
        return self._n_particles

    @property
    def lr(self):
        return self._lr

    @property
    def kappa(self):
        return self._kappa

    @property
    def empirical(self):
        return self._empirical

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_particles", type=int, default=51)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--kappa", type=float, default=1e-1)
    parser.add_argument("--dist_type", type=str, default="normal")
    parser.add_argument("--dist_params", type=str, default="0 1")
    parser.add_argument("--loss", type=str, default="quantile_huber")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_monte_carlo", type=int, default=1000)
    parser.add_argument("--style", type=str, default="seaborn")
    parser.add_argument("-i", type=int, default=30, help="Animation frame interval")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
