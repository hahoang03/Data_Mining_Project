import pandas as pd

# --- Load dataset ---
df = pd.read_csv('data/heart_disease.csv')

# --- Identify columns ---
numeric_features = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI',
                    'Sleep Hours', 'Triglyceride Level', 'Fasting Blood Sugar',
                    'CRP Level', 'Homocysteine Level']

categorical_features = ['Gender', 'Smoking', 'Diabetes',
                        'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol']

ordinal_features = ['Exercise Habits', 'Alcohol Consumption', 'Stress Level', 'Sugar Consumption']

# --- Check outlier numeric using IQR ---
print("=== Numeric Outliers (IQR method) ===")
outlier_summary = []

for col in numeric_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    outlier_summary.append({'Attribute': col, 'Outliers Count': len(outliers)})
    
outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df)

