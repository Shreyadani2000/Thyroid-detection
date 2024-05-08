import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

data = pd.read_csv('balanced_dataset (1).csv')

columns = ['Age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'Sex_M', 'Sick_t',
           'Pregnant_t', 'Thyroid Surgery_t', 'Goitre_t', 'Tumor_t']

X = data[columns]  
y = data['Category']  

# Spliting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)


# Evaluating Random Forest
rf_accuracy = accuracy_score(y_test, rf_predictions)


pickle.dump(rf_model, open('model.pkl', 'wb'))