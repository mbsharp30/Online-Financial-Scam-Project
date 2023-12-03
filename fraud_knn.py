import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import precision_recall_curve, auc, PrecisionRecallDisplay
import matplotlib.pyplot as plt

# Load the datasets
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')['isFraud']
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')['isFraud']

# Apply Stratified Sampling to Reduce Training Data Size
# Adjust the 'test_size' to change the proportion of data used for training (e.g., 0.25 for 25%)
X_train_sampled, _, y_train_sampled, _ = train_test_split(X_train, y_train, test_size=0.9, random_state=42)
X_test_sampled, _, y_test_sampled, _ = train_test_split(X_test, y_test, stratify=y_test, test_size=0.9, random_state=42)
# KNN Classifier
knn = KNeighborsClassifier()

# Hyperparameter tuning using Grid Search
param_grid = {
    'n_neighbors': [3, 4, 5, 6, 7],  # Example range
    'metric': ['euclidean', 'manhattan']  # Distance metrics
}

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1', verbose=1)
grid_search.fit(X_train_sampled, y_train_sampled)

# Best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Training the model with the best hyperparameters
best_knn = grid_search.best_estimator_
best_knn.fit(X_train_sampled, y_train_sampled)

# Predictions and Evaluation
predictions = best_knn.predict(X_test_sampled)

# Evaluation Metrics
precision = precision_score(y_test_sampled, predictions)
recall = recall_score(y_test_sampled, predictions)
f1 = f1_score(y_test_sampled, predictions)

# Output the Evaluation Metrics
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nClassification Report:")
print(classification_report(y_test_sampled, predictions))

# Predict probabilities for AUPRC
y_scores = best_knn.predict_proba(X_test_sampled)[:, 1]

# Compute Precision-Recall and plot AUPRC
precision, recall, thresholds = precision_recall_curve(y_test_sampled, y_scores)
auprc = auc(recall, precision)

# Plotting Precision-Recall Curve
plt.figure()
PrecisionRecallDisplay(precision=precision, recall=recall).plot()
plt.title(f'Precision-Recall Curve (AUPRC = {auprc:.2f})')
plt.show()