import argparse
import jax
import jax.numpy as jnp

def main(args):
    n = args.n
    n_paths = args.paths
    rng = jax.random.PRNGKey(0)

    for i in range(n_paths):
        rng, sub = jax.random.split(rng)
        t, b_t = brownian_points(sub, n)
        with open(f"brownian{n}{i+1}.dat", 'w') as f:
            for (s, b_s) in zip(t, b_t):
                f.write(f"{s}\t{b_s}\n")

def brownian_points(rng, n):
    loc = jnp.zeros(n)
    t = jnp.linspace(0,1,n)
    scale = jnp.clip(jax.vmap(lambda s: jnp.where(t < s, t, s))(t), 1e-7, 1.)
    return t, jax.random.multivariate_normal(rng, loc, scale)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int)
    parser.add_argument("--paths", type=int, default=1)
    args = parser.parse_args()
    main(args)
