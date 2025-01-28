import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = 'quicksave.csv'  # Change this to your actual file path
df = pd.read_csv(file_path)

# Convert columns to numeric
df = df.astype({'Step': int, 'Goals': int, 'Saves': int, 'Misses': int})

# Map every 10 steps into one
df['Aggregated Step'] = (df['Step'] // 10) * 10

# Aggregate data by Aggregated Step
grouped = df.groupby("Aggregated Step")[["Goals", "Saves"]].sum().reset_index()

# Compute save percentage
grouped["Save Percentage"] = (grouped["Saves"] / (grouped["Goals"] + grouped["Saves"])) * 100

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(grouped["Aggregated Step"], grouped["Save Percentage"], marker='o', linestyle='-', color='b', label="Save %")
plt.xlabel("Step")
plt.ylabel("Save Percentage")
plt.title("Evoulution of save percentage during traning")
#plt.legend()
plt.show()