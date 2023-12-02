import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('Fraud.csv')

# Preprocessing

# Drop the 'step' column
df.drop('step', axis=1, inplace=True)

# Feature Scaling - Z-score Standardization
numeric_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Custom Encoding for 'nameOrig' and 'nameDest'
df['customerTypeOrig'] = df['nameOrig'].apply(lambda x: 0 if x.startswith('C') else 1)
df['customerTypeDest'] = df['nameDest'].apply(lambda x: 0 if x.startswith('C') else 1)
df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

# Feature Extraction - Creating new features
df['balanceDiffOrig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']
df.drop(['newbalanceOrig', 'oldbalanceOrg', 'newbalanceDest', 'oldbalanceDest'], axis=1)

# One-Hot Encoding for 'type'
categorical_features = ['type']
one_hot = OneHotEncoder()
transformer = ColumnTransformer(transformers=[("OneHot", one_hot, categorical_features)], remainder='passthrough')

# Apply the transformation
df_transformed = transformer.fit_transform(df)

# Get the names of the one-hot encoded columns
one_hot_feature_names = transformer.named_transformers_['OneHot'].get_feature_names_out(categorical_features)

# Combine with the names of the untransformed columns
untransformed_columns = [col for col in df.columns if col not in categorical_features]
new_column_names = np.concatenate([one_hot_feature_names, untransformed_columns])

# Create a new DataFrame with these names
df_transformed = pd.DataFrame(df_transformed, columns=new_column_names)

# Train-Test Split
X = df_transformed.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = df['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversampling - Handling Imbalanced Dataset
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

# Save the train-test split datasets
X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
