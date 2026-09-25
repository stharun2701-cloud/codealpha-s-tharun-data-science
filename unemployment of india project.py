#!/usr/bin/env python
# coding: utf-8

# In[14]:


# TASK 2: Unemployment Analysis with Python

import pandas as pd
import matplotlib.pyplot as plt


# 1. Load Dataset


df = pd.read_csv("Unemployment in India.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)


# In[15]:



# 2. Data Cleaning


# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)


# In[16]:



# 3. Basic Statistics


print("\nStatistical Summary:")
print(df.describe())


# In[17]:



# 4. Monthly Average Unemployment Rate


monthly_unemployment = (
    df.groupby("Date")["Estimated Unemployment Rate (%)"]
    .mean()
    .reset_index()
)

print("\nMonthly Average Unemployment:")
print(monthly_unemployment)


# In[18]:



# 5. Plot Overall Unemployment Trend


plt.figure(figsize=(12, 6))

plt.plot(
    monthly_unemployment["Date"],
    monthly_unemployment["Estimated Unemployment Rate (%)"],
    marker="o"
)

plt.title("Unemployment Rate in India Over Time")
plt.xlabel("Date")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[19]:



# 6. COVID-19 Impact


df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

yearly_unemployment = (
    df.groupby("Year")["Estimated Unemployment Rate (%)"]
    .mean()
)

print("\nAverage Unemployment by Year:")
print(yearly_unemployment)


# In[20]:



# 7. Rural vs Urban Analysis


area_unemployment = (
    df.groupby("Area")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
)

print("\nRural vs Urban Unemployment:")
print(area_unemployment)

plt.figure(figsize=(8, 5))

area_unemployment.plot(kind="bar")

plt.title("Average Unemployment Rate: Rural vs Urban")
plt.xlabel("Area")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# In[21]:



# 8. State/Region Analysis


region_unemployment = (
    df.groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Unemployment by Region:")
print(region_unemployment)

# Top 10 regions
top_10 = region_unemployment.head(10)

plt.figure(figsize=(10, 6))

top_10.plot(kind="bar")

plt.title("Top 10 Regions by Average Unemployment Rate")
plt.xlabel("Region")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[22]:



# 9. COVID Period Analysis


covid_period = df[
    (df["Date"] >= "2020-03-01") &
    (df["Date"] <= "2020-06-30")
]

print("\nCOVID-19 Period Average Unemployment:")
print(
    covid_period["Estimated Unemployment Rate (%)"].mean()
)


# In[23]:



# 10. Highest and Lowest Monthly Unemployment


highest = monthly_unemployment.loc[
    monthly_unemployment["Estimated Unemployment Rate (%)"].idxmax()
]

lowest = monthly_unemployment.loc[
    monthly_unemployment["Estimated Unemployment Rate (%)"].idxmin()
]

print("\nHighest Monthly Unemployment:")
print(highest)

print("\nLowest Monthly Unemployment:")
print(lowest)


# In[24]:



# 11. Labour Participation Analysis


labour_participation = (
    df.groupby("Date")
    ["Estimated Labour Participation Rate (%)"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

plt.plot(
    labour_participation["Date"],
    labour_participation["Estimated Labour Participation Rate (%)"],
    marker="o"
)

plt.title("Labour Participation Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Labour Participation Rate (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[25]:



# 12. Final Summary


print("\n========== FINAL SUMMARY ==========")

print(
    "Highest unemployment month:",
    highest["Date"].strftime("%B %Y"),
    f"({highest['Estimated Unemployment Rate (%)']:.2f}%)"
)

print(
    "Lowest unemployment month:",
    lowest["Date"].strftime("%B %Y"),
    f"({lowest['Estimated Unemployment Rate (%)']:.2f}%)"
)

print(
    "Average unemployment during 2019:",
    df[df["Year"] == 2019]
    ["Estimated Unemployment Rate (%)"]
    .mean()
)

print(
    "Average unemployment during 2020:",
    df[df["Year"] == 2020]
    ["Estimated Unemployment Rate (%)"]
    .mean()
)

print(
    "Average rural unemployment:",
    area_unemployment.get("Rural")
)

print(
    "Average urban unemployment:",
    area_unemployment.get("Urban")
)


# In[ ]:




