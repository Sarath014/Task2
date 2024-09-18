import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a DataFrame with the Titanic data
data = {
    'PassengerId': [1, 2, 3, 4, 5],
    'Survived': [0, 1, 1, 0, 0],
    'Pclass': [3, 1, 3, 1, 3],
    'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley (Elizabeth Florence)', 'Heikkinen, Miss. Laina', 'Allen, Mr. William Henry', 'Moran, Mr. James'],
    'Sex': ['male', 'female', 'female', 'male', 'male'],
    'Age': [22.0, 38.0, 26.0, 35.0, 27.0],
    'SibSp': [1, 1, 0, 0, 0],
    'Parch': [0, 0, 0, 0, 0],
    'Ticket': ['A/4 51211', 'PC 17599', 'STON/P 7321', '113783', '330877'],
    'Fare': [7.25, 71.2833, 7.75, 8.05, 8.4583],
    'Cabin': ['C85', 'C125', 'Unknown', 'Unknown', 'Unknown'],
    'Embarked': ['S', 'C', 'S', 'S', 'Q']
}

df = pd.DataFrame(data)

# Data Cleaning
# Handle missing values (replace 'Unknown' with NaN)
df.replace('Unknown', np.nan, inplace=True)

# Fill missing values for 'Age' with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Convert categorical variables to appropriate data types
df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')

# Exploratory Data Analysis (EDA)
# Descriptive statistics
print(df.describe())

# Visualization
# Distribution of numerical variables
sns.histplot(df['Age'], bins=30)
plt.title('Distribution of Age')
plt.show()

# Relationship between categorical and numerical variables
sns.boxplot(x='Sex', y='Age', data=df)
plt.title('Age vs. Sex')
plt.show()

# Correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='viridis')
plt.title('Correlation Matrix')
plt.show()

# Survival Rates by Passenger Class
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rates by Passenger Class')
plt.show()
