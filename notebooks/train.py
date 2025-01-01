#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize, LabelEncoder
import pickle


df = pd.read_csv('../data/stroke-data.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')


hypertension_values = {
    0: "present",
    1: "absent"
}
df.hypertension = df.hypertension.map(hypertension_values)

heart_disease_values = {
    0: "present",
    1: "absent"
}
df.heart_disease = df.heart_disease.map(heart_disease_values)


df = df.dropna()

y = df.stroke
X = df.drop(columns=['stroke'])


X_full_train, X_test, y_full_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_full_train = X_full_train.reset_index(drop=True)
y_full_train = y_full_train.reset_index(drop=True)

X_train, X_val, y_train, y_val = train_test_split(
    X_full_train, y_full_train, test_size=0.25, stratify=y_full_train, random_state=42
)


X_full_train.shape, X_train.shape, X_val.shape, X_test.shape


y_full_train.shape, y_train.shape, y_val.shape, y_test.shape


y_full_train.value_counts(normalize=True)




global_stroke = y_full_train.mean()
round(global_stroke, 2)


# In[64]:


X_full_train.columns


# In[65]:


numerical = ['age', 'avg_glucose_level', 'bmi']

categorical = ['gender', 'hypertension', 'heart_disease', 'ever_married',
               'work_type', 'residence_type', 'smoking_status']


dv = DictVectorizer(sparse=False)


train_dicts = X_train[categorical + numerical].to_dict(orient='records')
val_dicts = X_val[categorical + numerical].to_dict(orient='records')
test_dicts =X_test[categorical + numerical].to_dict(orient='records')
full_train_dicts = X_full_train[categorical + numerical].to_dict(orient='records')

x_train = dv.fit_transform(train_dicts)
x_val = dv.transform(val_dicts)
x_test = dv.transform(test_dicts)
x_full_train = dv.transform(full_train_dicts)


# Train a Logistic Regression model for binary classification
# Encode the target variables
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_full_train)  # Encode full training target
y_test_encoded = label_encoder.transform(y_test)  # Encode test target

# Initialize the Logistic Regression model
logreg = LogisticRegression(
    max_iter=4000,
    class_weight='balanced',
    solver='lbfgs',
    random_state=42
)

# Train the model on the full training data
logreg.fit(x_full_train, y_train_encoded)

# Make predictions on the test dataset
y_pred_encoded = logreg.predict(x_test)
logreg_probs = logreg.predict_proba(x_test)[:, 1]  # Probability of the positive class

# Evaluate the model's performance on the test dataset
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print(f"Accuracy: {accuracy:.4f}")

# Display detailed classification report
print("Classification Report:")
print(classification_report(y_test_encoded, y_pred_encoded))

# Calculate and display ROC AUC for binary classification
y_test_binarized = label_binarize(y_test_encoded, classes=[0, 1])
auc_score = roc_auc_score(y_test_binarized, logreg_probs)
print(f"Binary ROC AUC Score: {auc_score:.4f}")


# Select a final model
# here we would save the best model which we would use for deployment

# In[114]:


with open('model.pkl', 'wb') as f_out:
    pickle.dump(logreg, f_out)


# In[115]:


f_out.close()

