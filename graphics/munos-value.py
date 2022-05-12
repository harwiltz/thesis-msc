import argparse
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

def main(args):
    gamma = args.y
    res = args.r
    xs = jnp.linspace(0,1,res)
    ys = jnp.where((2 * gamma ** (1-xs)) >= (gamma ** xs),
                   2 * gamma ** (1 - xs),
                   gamma ** xs)

    with open("munos-value.dat", "w") as f:
        for (x, y) in zip(xs, ys):
            f.write(f"{x}\t{y}\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=int, default=300, help="resolution")
    parser.add_argument("-y", type=float, default=0.3, help="discount factor")
    args = parser.parse_args()
    main(args)
