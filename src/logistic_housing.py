import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Download some data
california = fetch_california_housing()
print(california.data.shape)  # Features (rows, columns)
print(california.feature_names)  # List of feature names
print(california.target.shape)  # Target (housing prices)

# Convert to DataFrame
california_df = pd.DataFrame(data=california.data, columns=california.feature_names)

california_df['going_to_sel'] = np.where(california_df['owner'] == 'First Owner', 0, 1)


# Display the first few rows of the DataFrame
print(california_df.head())

# Features and target
# X = df[['year', 'selling_price', 'km_driven']]
# y = df['owner'].to_numpy()
