import matplotlib.pyplot as plt
import pandas as pd

interesting = ["T_out", "T_cur", "T_goal", "T_cur[0.5]"]

df = pd.read_csv("result.csv")
df = df[interesting + ["t"]]
df.plot(figsize=(10, 10), x="t")
plt.show()
