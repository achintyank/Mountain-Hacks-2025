import os
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Load the dataset
# Make sure the CSV file path is correct (put it in the same folder or give the right path)
Diabetes_data = pd.read_csv("diabetes_prediction_dataset_MH_Hacks.csv")

Diabetes_data = Diabetes_data[Diabetes_data['smoking_history'] != 'No Info'].reset_index(drop=True)
Diabetes_data = Diabetes_data[Diabetes_data['gender'] != 'Other'].reset_index(drop=True)
pd.set_option('display.max_rows', None) 

Diabetes_data.dropna(inplace=True)
Diabetes_data.info()

# Attempting to cast string columns to float will cause an error unless already encoded
# Let's encode 'gender' and 'smoking_history' properly first
Diabetes_data['gender_float'] = Diabetes_data['gender'].map({'Male': 1.0, 'Female': 0.0})
Diabetes_data['smoking_history_float'] = Diabetes_data['smoking_history'].astype('category').cat.codes.astype(float)

Diabetes_data['hypertension_float'] = Diabetes_data['hypertension'].astype(float)
Diabetes_data['heart_disease_float'] = Diabetes_data['heart_disease'].astype(float)
Diabetes_data['blood_glucose_level_float'] = Diabetes_data['blood_glucose_level'].astype(float)
Diabetes_data['diabetes_float'] = Diabetes_data['diabetes'].astype(float)

print(Diabetes_data.head())

# Feature Engineering
Diabetes_data['gender_value'] = Diabetes_data['gender_float'] * 20
Diabetes_data['age_value'] = Diabetes_data['age'] * 2
Diabetes_data['hypertension_value'] = Diabetes_data['hypertension_float'] * 20
Diabetes_data['heart_disease_value'] = Diabetes_data['heart_disease_float'] * 20
Diabetes_data['smoking_history_value'] = Diabetes_data['smoking_history_float'] * 10
Diabetes_data['bmi_value'] = Diabetes_data['bmi'] * 3
Diabetes_data['HbA1c_level_value'] = Diabetes_data['HbA1c_level'] * 20
Diabetes_data['blood_glucose_level_value'] = Diabetes_data['blood_glucose_level_float']

Diabetes_data['formula_value'] = (
    Diabetes_data['gender_value'] +
    Diabetes_data['age_value'] +
    Diabetes_data['hypertension_value'] +
    Diabetes_data['heart_disease_value'] +
    Diabetes_data['smoking_history_value'] +
    Diabetes_data['bmi_value'] +
    Diabetes_data['HbA1c_level_value'] +
    Diabetes_data['blood_glucose_level_value']
)
print(Diabetes_data.head())


X = Diabetes_data[['formula_value']]
y = Diabetes_data['diabetes']

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Function to predict diabetes based on user input
def predict_diabetes():
    print("\nEnter the following details to predict diabetes")
    
    # Collecting user input for each variable
    gender = input("Gender (Male = 1, Female = 0): ")
    age = input("Age: ")
    hypertension = input("Hypertension (0 = No, 1 = Yes): ")
    heart_disease = input("Heart Disease (0 = No, 1 = Yes): ")
    smoking_history = input("Smoking History (Never = 0, Ever = 1, Former = 2, Not Current = 3, Current = 4): ")
    bmi = input("BMI: ")
    HbA1c_level = input("HbA1c Level: ")
    blood_glucose_level = input("Blood Glucose Level: ")

    # Calculating formula_value based on user input
    formula_value = (
        float(gender) * 20 +
        float(age) * 2 +
        float(hypertension) * 20 +
        float(heart_disease) * 20 +
        float(smoking_history) * 10 +
        float(bmi) * 3 +
        float(HbA1c_level) * 20 +
        float(blood_glucose_level)
    )

    # Making a prediction using the trained model
    prediction = clf.predict([[formula_value]])[0]

    # Output the result
    if prediction == 0:
        print("\nPrediction: The person does NOT have diabetes.")
    else:
        print("\nPrediction: The person HAS diabetes.")

# Call the function to predict diabetes
predict_diabetes()
