import pandas as pd
import arff

# --- Load CSV ---
df = pd.read_csv('data/heart_smote_selected.csv', keep_default_na=False)

# --- Column definitions ---
target_col = 'Heart Disease Status'

# Numeric columns
numeric_cols = [
    'Age', 'Blood Pressure', 'Cholesterol Level', 'BMI',
    'Sleep Hours', 'Triglyceride Level', 'Fasting Blood Sugar',
    'CRP Level', 'Homocysteine Level'
]

# Binary
binary_cols = [
    'Gender', 'Smoking', 'Family Heart Disease', 'Diabetes',
    'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol'
]

# Ordinal
ordinal_cols = [
    'Exercise Habits', 'Alcohol Consumption', 'Stress Level', 'Sugar Consumption'
]

# Value mappings
ordinal_mapping = {
    'Exercise Habits': ['0','1','2'],
    'Alcohol Consumption': ['0','1','2','3'],
    'Stress Level': ['0','1','2'],
    'Sugar Consumption': ['0','1','2']
}

binary_mapping = ['0','1']
target_mapping = ['0','1']

# Convert nominal columns to string
for col in binary_cols + ordinal_cols + [target_col]:
    df[col] = df[col].astype(str).str.strip()

# Desired ARFF order EXACTLY as requested
ordered_cols = [
    'Age',
    'Gender',
    'Blood Pressure',
    'Cholesterol Level',
    'Exercise Habits',
    'Smoking',
    'Family Heart Disease',
    'Diabetes',
    'BMI',
    'High Blood Pressure',
    'Low HDL Cholesterol',
    'High LDL Cholesterol',
    'Alcohol Consumption',
    'Stress Level',
    'Sleep Hours',
    'Sugar Consumption',
    'Triglyceride Level',
    'Fasting Blood Sugar',
    'CRP Level',
    'Homocysteine Level',
    'Heart Disease Status'
]

# Reorder dataframe
df = df[ordered_cols]

# Build ARFF dictionary
arff_dict = {
    'description': 'Heart Disease Dataset',
    'relation': 'heart_transformed',
    'attributes': [],
    'data': df.values.tolist()
}

# Add attributes in correct order
for col in ordered_cols:

    if col in numeric_cols:
        arff_dict['attributes'].append((col, 'NUMERIC'))

    elif col in binary_cols:
        arff_dict['attributes'].append((col, binary_mapping))

    elif col in ordinal_cols:
        arff_dict['attributes'].append((col, ordinal_mapping[col]))

    elif col == target_col:
        arff_dict['attributes'].append((col, target_mapping))

# Save ARFF
with open('data/heart_improve.arff', 'w') as f:
    arff.dump(arff_dict, f)

print("ARFF file saved successfully with correct attribute order!")
