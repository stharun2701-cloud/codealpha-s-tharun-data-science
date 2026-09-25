#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Iris Flower Classification

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
df = pd.read_csv("iris.csv")


# In[2]:


# 2. Display basic information
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
print(df.info())

print("\nSpecies distribution:")
print(df["Species"].value_counts())


# In[3]:


# 3. Select features and target
X = df.drop(columns=["Id", "Species"])
y = df["Species"]


# In[4]:


# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# In[5]:


# 5. Create machine learning pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])


# In[6]:


# 6. Train the model
model.fit(X_train, y_train)


# In[7]:


# 7. Make predictions
y_pred = model.predict(X_test)


# In[9]:


# 8. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[10]:


# 9. Test with a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("\nNew Flower Prediction:")
print(prediction[0])


# In[ ]:




