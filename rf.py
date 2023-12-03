import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

class RandomForest:
    def __init__(self, nest, maxFeat, criterion, maxDepth, minSamplesLeaf):
        self.nest = nest
        self.maxFeat = maxFeat
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minSamplesLeaf = minSamplesLeaf
        self.model = {}

    def generate_bootstrap(self, xTrain, yTrain):
        n_samples = xTrain.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        xBoot = xTrain[indices]
        yBoot = yTrain[indices]
        oobIdx = np.setdiff1d(np.arange(n_samples), indices)
        return xBoot, yBoot, oobIdx

    def generate_subfeat(self, xTrain):
        n_features = xTrain.shape[1]
        maxFeat = min(self.maxFeat, n_features)
        featIdx = np.random.choice(n_features, size=maxFeat, replace=False)
        xSubfeat = xTrain[:, featIdx]
        return xSubfeat, featIdx

    def train(self, xTrain, yTrain):
        oob_errors = []
        for i in range(self.nest):
            xBoot, yBoot, oobIdx = self.generate_bootstrap(xTrain, yTrain)
            xSubfeat, featIdx = self.generate_subfeat(xBoot)
            
            tree = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.maxDepth, 
                                          min_samples_leaf=self.minSamplesLeaf)
            tree.fit(xSubfeat, yBoot)
            self.model[i] = {'tree': tree, 'feat': featIdx}

            if len(oobIdx) > 0:
                oob_predictions = self.predict(xTrain[oobIdx, :])
                oob_error = np.mean(yTrain[oobIdx] != oob_predictions)
                oob_errors.append(oob_error)

        return oob_errors

    def predict(self, xTest):
        predictions = np.zeros((xTest.shape[0], self.nest))
        for i, model in self.model.items():
            tree, featIdx = model['tree'], model['feat']
            predictions[:, i] = tree.predict(xTest[:, featIdx])
        return np.mean(predictions, axis=1) >= 0.5

    def predict_proba(self, xTest):
        probas = np.zeros((xTest.shape[0], self.nest))
        for i, model in self.model.items():
            tree, featIdx = model['tree'], model['feat']
            probas[:, i] = tree.predict_proba(xTest[:, featIdx])[:, 1]
        return np.mean(probas, axis=1)

# Load the datasets
X_train_full = pd.read_csv('X_train.csv')
y_train_full = pd.read_csv('y_train.csv')['isFraud']
X_test_full = pd.read_csv('X_test.csv')
y_test_full = pd.read_csv('y_test.csv')['isFraud']

# Sample 10% of the data
X_train = X_train_full.sample(frac=0.1, random_state=42).values
y_train = y_train_full.sample(frac=0.1, random_state=42).values
X_test = X_test_full.sample(frac=0.1, random_state=42).values
y_test = y_test_full.sample(frac=0.1, random_state=42).values

# Define the parameter grid
nest_values = [50]  # Reduced number of trees
maxFeat_values = [int(X_train.shape[1] * 0.3), int(X_train.shape[1] * 0.6), int(X_train.shape[1] * 0.9)]
criterion_values = ['gini', 'entropy']

# Initialize variables to store the best parameters and lowest error
best_params = None
lowest_error = float('inf')

# Grid search
for nest in nest_values:
    for maxFeat in maxFeat_values:
        for criterion in criterion_values:
            rf = RandomForest(nest, maxFeat, criterion, None, 1)  # Using default values for maxDepth and minSamplesLeaf
            oob_errors = rf.train(X_train, y_train)
            avg_oob_error = np.mean(oob_errors)
            print(f"OOB Error for params ({nest}, {maxFeat}, {criterion}): {avg_oob_error}")

            if avg_oob_error < lowest_error:
                lowest_error = avg_oob_error
                best_params = (nest, maxFeat, criterion, None, 1)

# Train the best model
best_rf = RandomForest(*best_params)
best_rf.train(X_train, y_train)

# Predictions and Evaluation
predictions = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Predict probabilities for AUPRC
y_scores = best_rf.predict_proba(X_test)

# Compute Precision-Recall and plot AUPRC
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
auprc = auc(recall, precision)
print(f"AUPRC: {auprc:.2f}")

# Plotting Precision-Recall Curve
plt.figure()
plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'2-class Precision-Recall curve: AUPRC={auprc:.2f}')
plt.show()

print(f"Best Parameters: {best_params}")
print(f"Lowest OOB Error: {lowest_error}")