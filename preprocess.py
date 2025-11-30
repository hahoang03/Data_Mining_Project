import pandas as pd
import numpy as np

# ==========================================
# 1. Load dataset
# ==========================================
print("=== LOADING DATA ===")
df = pd.read_csv("heart.csv")      
print("Dataset shape:", df.shape)
print(df.head())


# ==========================================
# 2. Missing Values
# ==========================================
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# fill hoặc drop:
df = df.dropna()    


# ==========================================
# 3. Remove Duplicates
# ==========================================
print("\n=== DUPLICATES ===")
duplicates = df.duplicated().sum()
print("Number of duplicated rows:", duplicates)

df = df.drop_duplicates()


# ==========================================
# 4. Outlier Handling (IQR)
#    Áp dụng cho các cột numeric
# ==========================================
print("\n=== OUTLIER REMOVAL (IQR) ===")

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    before = df.shape[0]
    df = df[(df[col] >= lower) & (df[col] <= upper)]
    after = df.shape[0]

    print(f"{col}: removed {before - after} outliers")


# ==========================================
# 5. One-Hot Encoding (chỉ tạo cột 0/1)
# ==========================================
print("\n=== ONE-HOT ENCODING ===")

categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
print("Categorical columns:", list(categorical_cols))

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("New dataset shape:", df.shape)


# ==========================================
# 6. Final Export
# ==========================================
output_path = "heart_cleaned.csv"
df.to_csv(output_path, index=False)

print("\n=== PREPROCESS COMPLETE ===")
print("Saved cleaned file to:", output_path)
print(df.head())
