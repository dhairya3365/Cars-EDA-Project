"""
Cars Exploratory Data Analysis (EDA) Project
Student: Dhairya Chovatiya

Dataset:
Vehicle Dataset from CarDekho - Kaggle
https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho

The assignment asks for data understanding, cleaning, descriptive analysis,
at least six meaningful charts using at least four chart types, and five
evidence-based findings. No machine-learning model is required.

The script first looks for the CSV in the current folder. If it is not found,
it downloads a public mirror of the same Kaggle dataset from GitHub.
"""

import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load data
# -----------------------------
LOCAL_FILE = Path("CAR DETAILS FROM CAR DEKHO.csv")
MIRROR_URL = (
    "https://raw.githubusercontent.com/bagassenop/cardekho/main/"
    "CAR%20DETAILS%20FROM%20CAR%20DEKHO.csv"
)

if LOCAL_FILE.exists():
    csv_path = LOCAL_FILE
else:
    print("Local CSV not found. Downloading the public CarDekho dataset...")
    csv_path = Path("CAR DETAILS FROM CAR DEKHO.csv")
    urlretrieve(MIRROR_URL, csv_path)

df = pd.read_csv(csv_path)

print("=" * 70)
print("CARS DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# -----------------------------
# 2. Dataset understanding
# -----------------------------
print("\n1. DATASET SHAPE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n2. COLUMN NAMES")
print(df.columns.tolist())

print("\n3. FIRST FIVE ROWS")
print(df.head())

print("\n4. DATA TYPES")
print(df.dtypes)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include="object").columns.tolist()

print("\nNumerical columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# -----------------------------
# 3. Data quality checks
# -----------------------------
print("\n5. MISSING VALUES")
print(df.isnull().sum())

print("\n6. DUPLICATE ROWS")
duplicate_count = df.duplicated().sum()
print("Exact duplicate rows:", duplicate_count)

print("\n7. UNIQUE CATEGORIES")
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

# -----------------------------
# 4. Cleaning
# -----------------------------
clean = df.copy()

# Standardise column names
clean.columns = clean.columns.str.strip().str.lower()

# Remove accidental leading/trailing spaces in text fields
for col in clean.select_dtypes(include="object").columns:
    clean[col] = clean[col].str.strip()

# Convert numerical columns to numeric safely
for col in ["year", "selling_price", "km_driven"]:
    clean[col] = pd.to_numeric(clean[col], errors="coerce")

# Keep all legitimate listings; do not blindly delete duplicates/outliers.
# Missing numeric rows are dropped only if conversion created missing values.
before_cleaning = len(clean)
clean = clean.dropna(subset=["year", "selling_price", "km_driven"])
after_cleaning = len(clean)

print("\n8. CLEANING SUMMARY")
print("Rows before cleaning:", before_cleaning)
print("Rows after basic cleaning:", after_cleaning)
print("Rows removed:", before_cleaning - after_cleaning)
print("Remaining missing values:")
print(clean.isnull().sum())

# Add useful derived variable
clean["car_age"] = 2020 - clean["year"]

# -----------------------------
# 5. Descriptive statistics
# -----------------------------
print("\n9. NUMERICAL DESCRIPTIVE STATISTICS")
print(clean[numeric_cols].describe().T)

print("\n10. CATEGORICAL DESCRIPTIVE STATISTICS")
for col in categorical_cols:
    print(f"\n{col}:")
    print(clean[col].value_counts())

# -----------------------------
# 6. Outlier identification
# -----------------------------
print("\n11. OUTLIER CHECK USING IQR")
for col in ["selling_price", "km_driven"]:
    q1 = clean[col].quantile(0.25)
    q3 = clean[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = ((clean[col] < lower) | (clean[col] > upper)).sum()
    print(
        f"{col}: Q1={q1:.2f}, Q3={q3:.2f}, "
        f"lower={lower:.2f}, upper={upper:.2f}, outliers={count}"
    )

# -----------------------------
# 7. Charts
# -----------------------------
sns.set_theme(style="whitegrid")

# Chart 1 - Fuel distribution
plt.figure(figsize=(8, 5))
order = clean["fuel"].value_counts().index
sns.countplot(data=clean, x="fuel", order=order)
plt.title("Distribution of Cars by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.show()

# Chart 2 - Seller type
plt.figure(figsize=(8, 5))
order = clean["seller_type"].value_counts().index
sns.countplot(data=clean, x="seller_type", order=order)
plt.title("Distribution of Cars by Seller Type")
plt.xlabel("Seller Type")
plt.ylabel("Number of Cars")
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()

# Chart 3 - Transmission
plt.figure(figsize=(7, 5))
order = clean["transmission"].value_counts().index
sns.countplot(data=clean, x="transmission", order=order)
plt.title("Manual vs Automatic Cars")
plt.xlabel("Transmission")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.show()

# Chart 4 - Owner
plt.figure(figsize=(9, 5))
order = clean["owner"].value_counts().index
sns.countplot(data=clean, x="owner", order=order)
plt.title("Distribution by Ownership History")
plt.xlabel("Owner Category")
plt.ylabel("Number of Cars")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# Chart 5 - Top brands
clean["brand"] = clean["name"].str.split().str[0]
top_brands = clean["brand"].value_counts().head(10)

plt.figure(figsize=(10, 6))
top_brands.sort_values().plot(kind="barh")
plt.title("Top 10 Car Brands by Number of Listings")
plt.xlabel("Number of Listings")
plt.ylabel("Brand")
plt.tight_layout()
plt.show()

# Chart 6 - Numeric correlation heatmap
corr = clean[["year", "selling_price", "km_driven"]].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Numerical Variables")
plt.tight_layout()
plt.show()

# Chart 7 - Selling price by owner
plt.figure(figsize=(9, 5))
sns.boxplot(data=clean, x="owner", y="selling_price", order=clean["owner"].value_counts().index)
plt.title("Selling Price by Ownership History")
plt.xlabel("Owner Category")
plt.ylabel("Selling Price (INR)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# Chart 8 - Year vs selling price
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=clean.sample(min(1500, len(clean)), random_state=42),
    x="year",
    y="selling_price",
    alpha=0.55
)
sns.regplot(
    data=clean.sample(min(1500, len(clean)), random_state=42),
    x="year",
    y="selling_price",
    scatter=False
)
plt.title("Vehicle Year vs Selling Price")
plt.xlabel("Vehicle Year")
plt.ylabel("Selling Price (INR)")
plt.tight_layout()
plt.show()

print("\n12. CORRELATIONS")
print(corr)

print("\n13. FINAL KEY FINDINGS")
print("1. Petrol and Diesel are the dominant fuel types in the dataset.")
print("2. Manual transmission cars are much more common than automatic cars.")
print("3. Individual sellers account for most listings.")
print("4. First-owner cars are the largest ownership group.")
print("5. Newer model years tend to have higher selling prices; year and price show a positive correlation.")
print("6. Kilometres driven has a weak negative relationship with selling price.")
print("7. Maruti and Hyundai are the two most frequently listed brands.")

print("\nEDA completed successfully.")
