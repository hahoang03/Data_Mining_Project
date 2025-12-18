import pandas as pd
from sklearn.preprocessing import StandardScaler

def check_missing(df, step_name):
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print(f"=== Check missing values {step_name} ===")
    if not missing.empty:
        print("Columns with missing values:")
        print(missing, "\n")
    else:
        print("No missing values.\n")
    return missing

# --- Step 0: Load dataset (keep_default_na=False so 'None' is not interpreted as NaN) ---
df = pd.read_csv('data/heart_filled.csv', skipinitialspace=True, keep_default_na=False)
print("=== Dataset loaded ===")
print(df.head(), "\n")
df.to_csv('step_0_loaded.csv', index=False)

# --- Step 1: Clean column names ---
df.columns = df.columns.str.strip()
print("=== Columns after cleaning ===")
print(df.columns.tolist(), "\n")
df.to_csv('step_1_columns_cleaned.csv', index=False)

# --- Step 2: Encode binary features ---
binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
binary_features = ['Gender', 'Smoking', 'Family Heart Disease', 'Diabetes',
                   'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol','Heart Disease Status']

for col in binary_features:
    if col in df.columns:
        df[col] = df[col].map(binary_map)

print("=== Dataset after encoding binary features ===")
print(df[binary_features].head(), "\n")
df.to_csv('step_2_binary_encoded.csv', index=False)
check_missing(df, "after binary encoding")

# --- Step 3: Encode ordinal features ---
ordinal_mapping = {
    'Exercise Habits': {'Low':0, 'Medium':1, 'High':2},
    'Alcohol Consumption': {'None':0, 'Low':1, 'Medium':2, 'High':3},
    'Stress Level': {'Low':0, 'Medium':1, 'High':2},
    'Sugar Consumption': {'Low':0, 'Medium':1, 'High':2}
}

for col, mapping in ordinal_mapping.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

print("=== Dataset after encoding ordinal features ===")
print(df[list(ordinal_mapping.keys())].head(), "\n")
df.to_csv('step_3_ordinal_encoded.csv', index=False)
check_missing(df, "after ordinal encoding")

# --- Step 4: Scale numeric features ---
numeric_features = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI',
                    'Sleep Hours', 'Triglyceride Level', 'Fasting Blood Sugar',
                    'CRP Level', 'Homocysteine Level']

numeric_features_existing = [col for col in numeric_features if col in df.columns]
scaler = StandardScaler()
df[numeric_features_existing] = scaler.fit_transform(df[numeric_features_existing])

print("=== Dataset after scaling numeric features ===")
print(df[numeric_features_existing].head(), "\n")
df.to_csv('step_4_numeric_scaled.csv', index=False)
check_missing(df, "after numeric scaling")

# --- Step 5: Save final transformed dataset (no feature selection) ---
df.to_csv('data/heart_preprocessed_before.csv', index=False)
print("Dataset saved to 'heart_preprocessed_before.csv'")