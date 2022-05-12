import argparse
import jax
import jax.numpy as jnp
import logging
import matplotlib.pyplot as plt

from functools import partial
from jax import vmap
from jax.scipy.stats.norm import pdf
from os.path import exists

LOC1 = -2.
SCALE1 = 0.5

LOC2 = 1.
SCALE2 = 1.

LAM = 0.4

XMIN = -4.
XMAX = 5.

RES = 250

PATH_PREFIX = "representations"
PATH_EXT = "dat"
MEAN_PATH = f"{PATH_PREFIX}-mean.{PATH_EXT}"
PDF_PATH = f"{PATH_PREFIX}-pdf.{PATH_EXT}"
CAT_PATH = f"{PATH_PREFIX}-cat.{PATH_EXT}"
QUANTILE_PATH = f"{PATH_PREFIX}-quantile.{PATH_EXT}"

def main(args):
    logger = logging.getLogger("main")
    N = args.n
    xs = jnp.linspace(XMIN, XMAX, RES)
    ys = (LAM * vmap(lambda x: pdf(x, LOC1, SCALE1))(xs)
          + (1 - LAM) * vmap(lambda x: pdf(x, LOC2, SCALE2))(xs))

    rng = jax.random.PRNGKey(0)
    rng, sub = jax.random.split(rng)

    ns = 10000
    choices = jax.random.bernoulli(sub, 1. - LAM, shape=(ns,))
    samples = jnp.where(choices == 0,
                        LOC1 + SCALE1 * jax.random.normal(rng, shape=(ns,)),
                        LOC2 + SCALE2 * jax.random.normal(rng, shape=(ns,)))

    if args.debug:
        plt.plot(xs, ys)
        plt.show()
        return

    write = args.f or (not exists(PDF_PATH))
    if (not write) and (not args.debug):
        logger.warning(f"File {PDF_PATH} already exists -- skipping. Use -f to overwrite it.")
    if write:
        with open(PDF_PATH, 'w') as f:
            for (x, y) in zip(xs, ys):
                f.write(f"{x}\t{y}\n")

    write = args.f or (not exists(MEAN_PATH))
    if (not write) and (not args.debug):
        logger.warning(f"File {MEAN_PATH} already exists -- skipping. Use -f to overwrite it.")
    if write:
        with open(MEAN_PATH, 'w') as f:
            f.write(f"{LAM * LOC1 + (1 - LAM) * SCALE1}\t{jnp.max(ys)}\n")

    write = args.f or (not exists(CAT_PATH))
    if (not write) and (not args.debug):
        logger.warning(f"File {CAT_PATH} already exists -- skipping. Use -f to overwrite it.")
    if write:
        with open(CAT_PATH, 'w') as f:
            atoms = jnp.linspace(XMIN, XMAX, N) + (XMAX-XMIN)/(2*N)
            cat_ys = vmap(lambda x: LAM * pdf(x, LOC1, SCALE1) + (1-LAM) * pdf(x, LOC2, SCALE2))(atoms)
            for (x, y) in zip(atoms, cat_ys):
                f.write(f"{x - (XMAX-XMIN)/(2*N)}\t{y}\n")

    write = args.f or (not exists(QUANTILE_PATH))
    if (not write) and (not args.debug):
        logger.warning(f"File {QUANTILE_PATH} already exists -- skipping. Use -f to overwrite it.")
    if write:
        with open(QUANTILE_PATH, 'w') as f:
            quantile_midpoints = (jnp.arange(N) + 0.5) / N
            quantiles = vmap(partial(jnp.quantile, samples))(quantile_midpoints)
            quantile_ys = jnp.ones_like(quantiles) / N
            for (x, y) in zip(quantiles, quantile_ys):
                f.write(f"{x}\t{y}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=11)
    parser.add_argument("-f", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()
    main(args)

