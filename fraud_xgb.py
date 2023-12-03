import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import precision_recall_curve, auc, PrecisionRecallDisplay
import matplotlib.pyplot as plt

# Load the datasets
X_train_full = pd.read_csv('X_train.csv')
y_train_full = pd.read_csv('y_train.csv')['isFraud']
X_test_full = pd.read_csv('X_test.csv')
y_test_full = pd.read_csv('y_test.csv')['isFraud']

# Apply Stratified Sampling to Reduce Training and Testing Data Size
# Sampling only 10% of the data
X_train, _, y_train, _ = train_test_split(X_train_full, y_train_full, test_size=0.9, stratify=y_train_full, random_state=42)
X_test, _, y_test, _ = train_test_split(X_test_full, y_test_full, test_size=0.9, stratify=y_test_full, random_state=42)

# Extreme Gradient Boosting Classifier
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Hyperparameter tuning using Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees, can be adjusted
    'max_depth': [3, 5, 7],          # Depth of trees, can be adjusted
    'learning_rate': [0.01, 0.1, 0.2] # Step size shrinkage used to prevent overfitting, can be adjusted
}

grid_search = GridSearchCV(xgb_clf, param_grid, cv=5, scoring='f1', verbose=1)
grid_search.fit(X_train, y_train)

# Best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Training the model with the best hyperparameters
best_xgb = grid_search.best_estimator_
best_xgb.fit(X_train, y_train)

# Predictions and Evaluation
predictions = best_xgb.predict(X_test)

# Evaluation Metrics
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

# Output the Evaluation Metrics
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Predict probabilities for AUPRC
y_scores = best_xgb.predict_proba(X_test)[:, 1]

# Compute Precision-Recall and plot AUPRC
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
auprc = auc(recall, precision)

# Plotting Precision-Recall Curve
plt.figure()
PrecisionRecallDisplay(precision=precision, recall=recall).plot()
plt.title(f'Precision-Recall Curve (AUPRC = {auprc:.2f})')
plt.show()
