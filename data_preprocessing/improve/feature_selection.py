import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# 1. Load dataset sau SMOTE
df = pd.read_csv("heart_smote.csv")

# 2. Tách feature và label
X = df.drop("Heart Disease Status", axis=1)
y = df["Heart Disease Status"]

# 3. Tính Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=42)

mi_df = pd.DataFrame({
    "Feature": X.columns,
    "MI Score": mi_scores
}).sort_values(by="MI Score", ascending=False)

# 4. In MI
print("\n=== MUTUAL INFORMATION SCORES ===")
print(mi_df)

# 5. Chọn feature MI > 0
selected_features = mi_df[mi_df["MI Score"] > 0]["Feature"].tolist()

# 6. Feature bị loại
removed_features = mi_df[mi_df["MI Score"] == 0]["Feature"].tolist()

# 7. Tạo dataset mới
df_selected = df[selected_features + ["Heart Disease Status"]]
df_selected.to_csv("heart_smote_selected.csv", index=False)

# 8. So sánh feature trước vs sau
print("\n=== FEATURE COMPARISON ===")
print(f"Original feature count : {X.shape[1]}")
print(f"Selected feature count : {len(selected_features)}")

print("\nRemoved features (MI = 0):")
for f in removed_features:
    print("-", f)

print("\nSelected features:")
for f in selected_features:
    print("-", f)

print("\nDataset saved as heart_smote_selected.csv")
