import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


TrainDF = pd.read_csv('Encoded_X_Train_20_percent_data.csv')
ValDF = pd.read_csv('Encoded_X_val_20_percent_data.csv')

TrainDF.head()

np.random.seed(42)


# Assign X and y
TrainDF_y = TrainDF['click']
ValDF_y = ValDF['click']

TrainDF_X =  TrainDF.drop(['click'], axis = 1)
ValDF_X =  ValDF.drop(['click'], axis = 1)

# A function for defining log loss
def calculate_log_loss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

#################### Baseline model & Log Loss ##############################
# Define the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest classifier
rf_classifier.fit(TrainDF_X, TrainDF_y)

# Make predictions on the train and validation set
train_pred_probs = rf_classifier.predict_proba(TrainDF_X)[:, 1]
val_pred_probs = rf_classifier.predict_proba(ValDF_X)[:, 1]

# Calculate log loss on the train and validation set
train_log_loss = calculate_log_loss(TrainDF_y, train_pred_probs)
val_log_loss = calculate_log_loss(ValDF_y, val_pred_probs)

# Print out results
print(f"Log Loss on Train Set: {train_log_loss}")
print(f"Log Loss on Val Set: {val_log_loss}")


#################### Additional Performance Metrics ############################
# Put the threshold to 0.5; predict "click" if greater than 0.5
train_predictions = (train_pred_probs >= 0.5).astype(int)
val_predictions = (val_pred_probs >= 0.5).astype(int)

def prediction_statistics(y_actual, y_predicted):
    """
    Method outputs
      - classification_report
      - accuracy & error scores
      - confusion matrix
    """
    feature_names = np.array(['Not Click','Click'])
    print(classification_report(y_actual, y_predicted, target_names=feature_names))
    accuracy = accuracy_score(y_actual, y_predicted) * 100
    error = 100-accuracy
    f1 = f1_score(y_actual, y_predicted)
    print(f"Accuracy: {round(accuracy,3)}%")
    print(f"Error: {round(error,3)}%\n")
    print(f"F1: {round(f1, 3)}%\n")

prediction_statistics(TrainDF_y,train_predictions)
prediction_statistics(ValDF_y,val_predictions)

train_f1 = f1_score(TrainDF_y, train_predictions)
print(train_f1) # 0.95

val_f1 = f1_score(ValDF_y, val_predictions)
print(val_f1) # 0.26

