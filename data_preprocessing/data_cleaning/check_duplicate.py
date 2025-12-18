import pandas as pd

# Load your CSV file
df = pd.read_csv("data/heart.csv")

# Check duplicate rows
duplicates = df[df.duplicated()]

# Count duplicates
print("\nNumber of duplicate rows:", df.duplicated().sum())
