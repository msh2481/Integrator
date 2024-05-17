import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
from mpl_toolkits.mplot3d import Axes3D  # type: ignore

df = pd.read_csv("result.csv")
columns = list(df.columns)
for c in columns:
    if "[" in c:
        df = df.drop(c, axis=1)
# for i in range(3):
#     old = "g" + "'" * i
#     new = "g" + "'" * (i + 1)
#     df[new] = (df[old].diff() / df["t"].diff()).clip(-df[old], df[old])
#     df[new] = df[new].rolling(5).mean()
df.plot(figsize=(10, 10), x="t")

# plt.figure(figsize=(10, 10))
# plt.plot(df["a"], df["b"], "k", lw=1)
# plt.axis("off")
plt.tight_layout()
plt.show()
# plt.savefig("a.svg")
