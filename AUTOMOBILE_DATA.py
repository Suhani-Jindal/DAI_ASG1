import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset('mpg')  # Inbuilt automobile dataset

# Display basic information
print(df.info())
print(df.head()) 


# 1--> PERFORMING DATA CLEANING

# Count missing values
print(df.isnull().sum())

# Fill missing values in 'horsepower' with median
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

# Drop rows with missing values (if necessary)
df.dropna(inplace=True)

# Verify again
print(df.isnull().sum())

# Count duplicates
print("Duplicate Rows:", df.duplicated().sum())

# Drop duplicates
df = df.drop_duplicates()

#Boxplot to find outliers in horsepower
sns.boxplot(x=df['horsepower'])
plt.title("Boxplot of Horsepower")
plt.show()

#removing extreme outliers
Q1 = df['horsepower'].quantile(0.25)
Q3 = df['horsepower'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier range
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter data
df = df[(df['horsepower'] >= lower_bound) & (df['horsepower'] <= upper_bound)]

# Convert 'origin' column to categorical type
df['origin'] = df['origin'].astype('category')

# Check unique values
print(df['origin'].unique())



# 2-->Exploratory Data Analysis (EDA):


# 2--> (i)--> Univariate Analysis (Single-Variable Exploration)
print(df.describe())  # Summary stats for numerical data
print(df['origin'].value_counts())  # Frequency distribution for categorical variable

# Histogram of MPG (Fuel Efficiency)
sns.histplot(df['mpg'], bins=20, kde=True)
plt.title("Distribution of Fuel Efficiency (MPG)")
plt.show()

# Boxplot for detecting outliers in car weight
sns.boxplot(x=df['weight'])
plt.title("Boxplot of Car Weights")
plt.show()


# 2--> (ii)-->Bivariate Analysis (Two-Variable Exploration)

#Scatter Plot (MPG vs Weight)
sns.scatterplot(x=df['weight'], y=df['mpg'])
plt.title("MPG vs Weight")
plt.show()

#Parallel Boxplot (MPG Across Origins)
sns.boxplot(x=df['origin'], y=df['mpg'])
plt.title("MPG Across Different Car Origins")
plt.show()


# 2--> (iii)--> Multivariate Analysis (Multiple Variables Exploration)

#Pair Plot (Multiple Relationships)
sns.pairplot(df[['mpg', 'horsepower', 'weight', 'acceleration']], diag_kind='kde')
plt.show()

#Heatmap of correaltion
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Features")
plt.show()

# Cluster Analysis (Car Categories Based on MPG & Weight)
from sklearn.cluster import KMeans
X = df[['horsepower', 'weight', 'mpg']]
X = X.dropna()
# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)
# Visualize Clusters
sns.scatterplot(x=df['weight'], y=df['mpg'], hue=df['cluster'], palette='coolwarm')
plt.title("Clusters of Cars Based on MPG and Weight")
plt.show()

