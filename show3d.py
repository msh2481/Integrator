import argparse

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore
from mpl_toolkits.mplot3d import Axes3D  # type: ignore

parser = argparse.ArgumentParser(description="3D plot from CSV data")
parser.add_argument("x", type=str, help="Column for x-axis")
parser.add_argument("y", type=str, help="Column for y-axis")
parser.add_argument("z", type=str, help="Column for z-axis")
parser.add_argument("--lw", type=float, default=1.0, help="Line width for the plot")
args = parser.parse_args()

df = pd.read_csv("result.csv")

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.plot(df[args.x], df[args.y], df[args.z], "k", lw=args.lw)

ax.set_xlabel(args.x)
ax.set_ylabel(args.y)
ax.set_zlabel(args.z)  # type: ignore
ax.legend()

plt.tight_layout()
plt.show()
