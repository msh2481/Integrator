import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("result.csv")
df.plot(figsize=(10, 10), x="t")
plt.show()
