import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file

df = pd.read_csv("csv.csv")

# Plot the data

id_where_1 = df[df["id"] == 1]

plt.plot(id_where_1["x"], label="x")
plt.plot(id_where_1["y"], label="y")
plt.plot(id_where_1["z"], label="z")
plt.legend()
plt.show()