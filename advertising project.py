#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Advertising Sales Prediction Project

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Step 1: Load and clean data
df = pd.read_csv("Advertising.csv")
df = df.drop(columns=["Unnamed: 0"])  # remove index column

# Step 2: Exploratory Data Analysis
print("Correlation Matrix:\n", df.corr())

# Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df)
plt.show()

# Step 3: Regression Model
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:", dict(zip(X.columns, model.coef_)))

# Step 4: Evaluate Model
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))

# RMSE calculation (manual sqrt for compatibility)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

# Residual Plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Actual vs Predicted Plot
plt.scatter(y_test, y_pred, color="blue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Step 5: Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_,
    "AbsImpact": np.abs(model.coef_)
}).sort_values(by="AbsImpact", ascending=False)
print("\nFeature Importance:\n", importance)

sns.barplot(x="Feature", y="AbsImpact", data=importance)
plt.title("Feature Importance (Impact on Sales)")
plt.show()

# Step 6: Forecast Future Sales (Scenario Simulation)
scenarios = pd.DataFrame({
    "TV": [200, 150, 300],
    "Radio": [30, 20, 40],
    "Newspaper": [20, 10, 5]
})

predicted_sales = model.predict(scenarios)
scenarios["Predicted_Sales"] = predicted_sales
print("\nScenario Predictions:\n", scenarios)

# Step 7: Actionable Insights
print("\nInsights:")
print("- TV spend has the strongest impact on sales.")
print("- Radio spend also contributes significantly.")
print("- Newspaper spend shows minimal effect; reallocating budget to TV/Radio may yield better ROI.")
print("- Combining TV + Radio campaigns tends to maximize sales outcomes.")


# In[ ]:




