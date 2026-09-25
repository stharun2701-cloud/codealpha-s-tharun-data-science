#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load the data
df = pd.read_csv("car data.csv")

# Clean column names
df.columns = df.columns.str.strip()

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# Remove duplicate records
df = df.drop_duplicates()


# Create a new feature for car age
df["Car_Age"] = 2026 - df["Year"]
df.drop("Year", axis=1, inplace=True)

print("\nData after cleaning:")
print(df.head())


# Create a folder to save all charts
os.makedirs("car_analysis_charts", exist_ok=True)


# ---------------------------------------------------------
# Exploratory Data Analysis
# ---------------------------------------------------------

# 1. Present Price vs Selling Price
plt.figure(figsize=(8, 5))
plt.scatter(df["Present_Price"], df["Selling_Price"], alpha=0.6)
plt.xlabel("Present Price")
plt.ylabel("Selling Price")
plt.title("Present Price vs Selling Price")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("car_analysis_charts/present_vs_selling.png", dpi=300)
plt.show()


# 2. Driven Kilometers vs Selling Price
plt.figure(figsize=(8, 5))
plt.scatter(df["Driven_kms"], df["Selling_Price"], alpha=0.6)
plt.xlabel("Driven Kilometers")
plt.ylabel("Selling Price")
plt.title("Driven Kilometers vs Selling Price")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("car_analysis_charts/kms_vs_selling.png", dpi=300)
plt.show()


# 3. Car Age vs Selling Price
plt.figure(figsize=(8, 5))
plt.scatter(df["Car_Age"], df["Selling_Price"], alpha=0.6)
plt.xlabel("Car Age")
plt.ylabel("Selling Price")
plt.title("Car Age vs Selling Price")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("car_analysis_charts/age_vs_selling.png", dpi=300)
plt.show()


# 4. Average selling price by fuel type
fuel_prices = df.groupby("Fuel_Type")["Selling_Price"].mean()

plt.figure(figsize=(7, 5))
bars = plt.bar(fuel_prices.index, fuel_prices.values)

plt.xlabel("Fuel Type")
plt.ylabel("Average Selling Price")
plt.title("Average Selling Price by Fuel Type")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.2f}",
        ha="center",
        va="bottom"
    )

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("car_analysis_charts/fuel_type.png", dpi=300)
plt.show()


# 5. Average selling price by transmission
transmission_prices = df.groupby("Transmission")["Selling_Price"].mean()

plt.figure(figsize=(7, 5))
bars = plt.bar(
    transmission_prices.index,
    transmission_prices.values
)

plt.xlabel("Transmission")
plt.ylabel("Average Selling Price")
plt.title("Average Selling Price by Transmission")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.2f}",
        ha="center",
        va="bottom"
    )

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("car_analysis_charts/transmission.png", dpi=300)
plt.show()


# 6. Average selling price by previous owners
owner_prices = df.groupby("Owner")["Selling_Price"].mean()

plt.figure(figsize=(7, 5))
bars = plt.bar(
    owner_prices.index.astype(str),
    owner_prices.values
)

plt.xlabel("Previous Owners")
plt.ylabel("Average Selling Price")
plt.title("Average Selling Price by Previous Owners")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.2f}",
        ha="center",
        va="bottom"
    )

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("car_analysis_charts/owners.png", dpi=300)
plt.show()


# ---------------------------------------------------------
# Prepare data for machine learning
# ---------------------------------------------------------

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

categorical_cols = X.select_dtypes(include="object").columns
numeric_cols = X.select_dtypes(exclude="object").columns

print("\nCategorical columns:")
print(list(categorical_cols))

print("\nNumerical columns:")
print(list(numeric_cols))


# Convert categorical columns into numerical values
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        )
    ],
    remainder="passthrough"
)


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ---------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------

linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_pred)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
linear_r2 = r2_score(y_test, linear_pred)

print("\nLinear Regression")
print("MAE:", round(linear_mae, 3))
print("RMSE:", round(linear_rmse, 3))
print("R2 Score:", round(linear_r2, 3))


# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestRegressor(
            n_estimators=300,
            random_state=42
        )
    )
])

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest")
print("MAE:", round(rf_mae, 3))
print("RMSE:", round(rf_rmse, 3))
print("R2 Score:", round(rf_r2, 3))


# ---------------------------------------------------------
# Compare the models
# ---------------------------------------------------------

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [linear_mae, rf_mae],
    "RMSE": [linear_rmse, rf_rmse],
    "R2 Score": [linear_r2, rf_r2]
})

print("\nModel comparison:")
print(results)


# Model comparison chart
plt.figure(figsize=(8, 5))

bars = plt.bar(
    results["Model"],
    results["R2 Score"]
)

plt.ylabel("R2 Score")
plt.title("Model Performance Comparison")
plt.ylim(0, 1)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.3f}",
        ha="center",
        va="bottom"
    )

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("car_analysis_charts/model_comparison.png", dpi=300)
plt.show()


# ---------------------------------------------------------
# Actual vs Predicted values
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    rf_pred,
    alpha=0.7
)

low = min(y_test.min(), rf_pred.min())
high = max(y_test.max(), rf_pred.max())

plt.plot(
    [low, high],
    [low, high],
    linestyle="--"
)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "car_analysis_charts/actual_vs_predicted.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# Residual analysis
# ---------------------------------------------------------

errors = y_test - rf_pred

plt.figure(figsize=(8, 5))

plt.scatter(
    rf_pred,
    errors,
    alpha=0.7
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted Price")
plt.ylabel("Prediction Error")
plt.title("Residual Analysis")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "car_analysis_charts/residual_analysis.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

rf = rf_model.named_steps["model"]
processor = rf_model.named_steps["preprocessor"]

feature_names = processor.get_feature_names_out()

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop 10 important features:")
print(importance.head(10))


plt.figure(figsize=(9, 6))

top = importance.head(10)

plt.barh(
    top["Feature"][::-1],
    top["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Features Used by Random Forest")

plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "car_analysis_charts/feature_importance.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# Predict price for a new car
# ---------------------------------------------------------

new_car = pd.DataFrame({
    "Car_Name": ["swift"],
    "Present_Price": [6.5],
    "Driven_kms": [30000],
    "Fuel_Type": ["Petrol"],
    "Selling_type": ["Dealer"],
    "Transmission": ["Manual"],
    "Owner": [0],
    "Car_Age": [4]
})

predicted_price = rf_model.predict(new_car)[0]

print("\n-----------------------------")
print("New Car Prediction")
print("-----------------------------")
print("Car:", new_car["Car_Name"].iloc[0])
print("Present Price:", new_car["Present_Price"].iloc[0])
print("Driven Kms:", new_car["Driven_kms"].iloc[0])
print("Fuel Type:", new_car["Fuel_Type"].iloc[0])
print("Transmission:", new_car["Transmission"].iloc[0])
print("Car Age:", new_car["Car_Age"].iloc[0])
print("Predicted Selling Price:", round(predicted_price, 2))


# ---------------------------------------------------------
# Create a simple prediction image
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))
plt.axis("off")

plt.text(
    0.5,
    0.85,
    "CAR PRICE PREDICTION",
    fontsize=22,
    fontweight="bold",
    ha="center"
)

details = (
    f"Car: {new_car['Car_Name'].iloc[0]}\n"
    f"Present Price: {new_car['Present_Price'].iloc[0]}\n"
    f"Driven Kms: {new_car['Driven_kms'].iloc[0]}\n"
    f"Fuel Type: {new_car['Fuel_Type'].iloc[0]}\n"
    f"Transmission: {new_car['Transmission'].iloc[0]}\n"
    f"Car Age: {new_car['Car_Age'].iloc[0]} years"
)

plt.text(
    0.25,
    0.60,
    details,
    fontsize=13,
    va="top"
)

plt.text(
    0.70,
    0.55,
    f"{predicted_price:.2f}",
    fontsize=28,
    fontweight="bold",
    ha="center"
)

plt.text(
    0.70,
    0.45,
    "Predicted Selling Price",
    fontsize=12,
    ha="center"
)

plt.savefig(
    "car_analysis_charts/car_price_prediction.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# Create one dashboard image
# ---------------------------------------------------------

fig, axes = plt.subplots(
    3,
    3,
    figsize=(17, 14)
)

fig.suptitle(
    "Car Price Prediction Analysis",
    fontsize=22,
    fontweight="bold"
)


# Present price
axes[0, 0].scatter(
    df["Present_Price"],
    df["Selling_Price"],
    alpha=0.5
)
axes[0, 0].set_title("Present Price vs Selling Price")
axes[0, 0].set_xlabel("Present Price")
axes[0, 0].set_ylabel("Selling Price")


# Kilometers
axes[0, 1].scatter(
    df["Driven_kms"],
    df["Selling_Price"],
    alpha=0.5
)
axes[0, 1].set_title("Driven Kms vs Selling Price")
axes[0, 1].set_xlabel("Driven Kms")
axes[0, 1].set_ylabel("Selling Price")


# Car age
axes[0, 2].scatter(
    df["Car_Age"],
    df["Selling_Price"],
    alpha=0.5
)
axes[0, 2].set_title("Car Age vs Selling Price")
axes[0, 2].set_xlabel("Car Age")
axes[0, 2].set_ylabel("Selling Price")


# Fuel
axes[1, 0].bar(
    fuel_prices.index,
    fuel_prices.values
)
axes[1, 0].set_title("Average Price by Fuel Type")
axes[1, 0].set_ylabel("Selling Price")


# Transmission
axes[1, 1].bar(
    transmission_prices.index,
    transmission_prices.values
)
axes[1, 1].set_title("Average Price by Transmission")
axes[1, 1].set_ylabel("Selling Price")


# Owners
axes[1, 2].bar(
    owner_prices.index.astype(str),
    owner_prices.values
)
axes[1, 2].set_title("Average Price by Previous Owners")
axes[1, 2].set_xlabel("Owners")
axes[1, 2].set_ylabel("Selling Price")


# Actual vs predicted
axes[2, 0].scatter(
    y_test,
    rf_pred,
    alpha=0.6
)

axes[2, 0].plot(
    [low, high],
    [low, high],
    linestyle="--"
)

axes[2, 0].set_title(
    f"Actual vs Predicted (R2 = {rf_r2:.3f})"
)

axes[2, 0].set_xlabel("Actual")
axes[2, 0].set_ylabel("Predicted")


# Feature importance
top_features = importance.head(8)

axes[2, 1].barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

axes[2, 1].set_title("Important Features")
axes[2, 1].set_xlabel("Importance")


# Model information
axes[2, 2].axis("off")

axes[2, 2].text(
    0.5,
    0.80,
    "RANDOM FOREST RESULTS",
    fontsize=15,
    fontweight="bold",
    ha="center"
)

axes[2, 2].text(
    0.5,
    0.55,
    f"MAE: {rf_mae:.3f}\n\n"
    f"RMSE: {rf_rmse:.3f}\n\n"
    f"R2 Score: {rf_r2:.3f}\n\n"
    f"Predicted Price: {predicted_price:.2f}",
    fontsize=13,
    ha="center",
    va="center"
)


plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    "car_analysis_charts/car_price_dashboard.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\n====================================")
print("PROJECT COMPLETED")
print("====================================")
print("All charts are saved in:")
print("car_analysis_charts")
print("\nRandom Forest R2 Score:", round(rf_r2, 3))
print("Predicted price:", round(predicted_price, 2))


# In[ ]:




