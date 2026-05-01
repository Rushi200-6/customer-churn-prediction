import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#  Load dataset
df = pd.read_csv("data.csv")

# Basic info
print("First 5 rows:")
print(df.head(11))

print("\nDataset Info:")
print(df.info())

print("\nStatistics:")
print(df.describe())

# Churn distribution
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

#  Tenure vs Churn
sns.boxplot(x="Churn", y="tenure", data=df)
plt.title("Tenure vs Churn")
plt.show()

# Monthly Charges vs Churn
sns.boxplot(x="Churn", y="monthly_charges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# Correlation heatmap
corr = df.corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
