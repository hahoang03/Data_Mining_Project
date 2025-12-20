# Handling an Imbalanced Heart Disease Dataset using Data Mining Techniques

## Team Members
**Group H&H**
- Nguyen Tran Hoang Ha – ITITIU21127  
- Nguyen Danh Huy – ITITIU22071  

---

## Project Overview
This project was conducted as part of the **Data Mining (IT160IU)** course at **Vietnam National University – Ho Chi Minh City, International University**.  
The main purpose of this study is to analyze and handle class imbalance in a heart disease dataset using data mining techniques, and to evaluate how data balancing affects classification performance.

The dataset includes multiple health, lifestyle, and medical risk factors. Due to class imbalance, traditional classifiers tend to favor the majority class, leading to misleading accuracy results. This project applies **SMOTE (Synthetic Minority Oversampling Technique)** to address this issue.

---

## Objectives
- Perform data preprocessing on a heart disease dataset  
- Handle missing values, duplicates, and outliers  
- Transform data using normalization and encoding  
- Address class imbalance using SMOTE  
- Apply and compare multiple classification algorithms  
- Evaluate models using appropriate performance metrics  

---

## Dataset
- **Source**: Kaggle – Heart Disease Dataset (by Oktay Rdeki)  
- **Origin**: Derived from the UCI Cleveland Heart Disease Dataset  
- **Number of instances**: 10,000  
- **Number of attributes**: 21  
- **Target variable**: Heart Disease Status (Yes / No)

The dataset contains:
- Numeric attributes (Age, Blood Pressure, Cholesterol Level, BMI, etc.)
- Binary attributes (Gender, Smoking, Diabetes, etc.)
- Ordinal attributes (Exercise Habits, Alcohol Consumption, Stress Level, etc.)

---

## Data Preprocessing

### Data Cleaning
- Missing values were detected and handled based on variable types:
  - Numeric variables: mean imputation  
  - Binary variables: mode imputation  
  - Ordinal variables: converted to numeric order and filled using median  
- Duplicate records were checked using `df.duplicated()` (no duplicates found)  
- Outliers were detected using the IQR method (no significant outliers detected)

### Data Transformation
- **Normalization**: Min–Max normalization was applied to numeric attributes  
- **Encoding categorical variables**:
  - Binary variables were encoded as 0 and 1  
  - Ordinal variables were encoded as: None = 0, Low = 1, Medium = 2, High = 3  
- **Feature Selection**:
  - Mutual Information was used to measure the relationship between features and the target variable

---

## Handling Class Imbalance
The original dataset is highly imbalanced, causing classifiers to predict only the majority class and resulting in high accuracy but very low Kappa values.

To address this problem, **SMOTE** was applied to generate synthetic samples for the minority class. This helps classifiers learn meaningful patterns instead of relying on class distribution.

---

## Classification Models
The following classifiers were implemented using **WEKA**:
- J48 Decision Tree  
- Naive Bayes  
- Random Forest  
- SMO (Support Vector Machine)

All models were evaluated using **Stratified 10-Fold Cross-Validation**.

---

## Evaluation Metrics
The models were evaluated using:
- Accuracy  
- F1-Score  
- Kappa Statistic  
- Runtime  

Kappa Statistic and F1-Score were emphasized since accuracy alone is not reliable for imbalanced datasets.

---

## Results Summary
- Before applying SMOTE:
  - Accuracy was around 80% for most classifiers
  - Kappa value was close to 0, indicating majority-class bias
- After applying SMOTE:
  - All classifiers achieved positive Kappa values
  - **Random Forest** achieved the best performance:
    - Accuracy: 83.38%
    - F1-Score: 0.8326
    - Kappa: 0.6675

---

## Tools and Technologies
- Python (Pandas, Scikit-learn)
- WEKA
- Jupyter Notebook
- Git & GitHub

---


## References
Please refer to the References section in the project report for detailed academic sources related to SMOTE, heart disease prediction, and machine learning techniques.
