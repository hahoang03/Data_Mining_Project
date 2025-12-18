import pandas as pd
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("data/heart_transformed.csv")

# Tách feature và label
X = df.drop("Heart Disease Status", axis=1)
y = df["Heart Disease Status"]

print("Before SMOTE:")
print(y.value_counts())

# Áp dụng SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nAfter SMOTE:")
print(y_resampled.value_counts())

# Gộp lại thành dataframe mới
df_smote = pd.concat(
    [
        pd.DataFrame(X_resampled, columns=X.columns),
        pd.Series(y_resampled, name="Heart Disease Status")
    ],
    axis=1
)

# Lưu file mới
df_smote.to_csv("heart_smote.csv", index=False)

print("\nSMOTE dataset saved as heart_smote.csv")
