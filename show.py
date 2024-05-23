import matplotlib.pyplot as plt
import pandas as pd  # type: ignore

df = pd.read_csv("result.csv")
columns = list(df.columns)
for c in columns:
    if "[" in c:
        df = df.drop(c, axis=1)
df.plot(figsize=(10, 10), x="t")

plt.tight_layout()
plt.show()
