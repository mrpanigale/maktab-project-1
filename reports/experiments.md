# Dataset:
- #### name = creditcards.csv => made by kaggle
- #### rows = 284,807
- #### features = 30 => time , v1 to v28 (PCA transformed),amount
- #### classes = binary => Fraud , Legitimate
- #### Imbalance = 0.017% Fraud , Rest% Legitimate
- #### Scaling = StandarScaler
- #### Train/test split = 80/20
- #### Test set= only for final test 

# Experiment KNN(Too Slow):
- #### scale = True
- #### metrics = F1,Recall ,precision,Confusion Matrix
- #### Best K = 5 
- #### Tested K = 1,5,20
- #### Best Thresholds = 0.3 , Because of detecting Fraudental transactions and Trade-off between FP,FN
- #### threshold scores for validation ==>  Precision: 89,Recall: 85
- #### Test data scores : recall = 75, precision= 90

# Experiment Tree:
- #### scale = No needed
- #### metrics = F1,Recall ,precision,Confusion Matrix
- #### Best max_depth = 5
- #### Tested max_depths = 2,5,10,None(Free to Grow)
- #### Best Thresholds = 0.3 , Because of detecting Fraudental transactions and Trade-off between FP,FN
- #### threshold scores for validation ==>  Precision: 91,Recall: 82
- #### Test data score:  Recall = 63,Precision = 88

# Experiment Logistic Regression:
- #### scale = True
- #### metrics = F1,Recall ,precision,Confusion Matrix
- #### Best C = 10
- #### Tested C = 0.01 , 0.1 , 1, 10  Bigger C => smaller penalty
- #### Best Thresholds = 0.3 , Because of detecting Fraudental transactions and Trade-off between FP,FN
- #### threshold scores for validation ==>  Precision: 86,Recall: 69
- #### Test data scores: Recall = 60,Precision = 82

# Experiment MLP(Big Capacity and faster than KNN):
- #### scale = True
- #### metrics = F1,Recall ,precision,Confusion Matrix
- #### Architecture:
  - ##### Input: 30 features
  - ##### Hidden layer 1: 64 neurons + ReLU + Dropout(0.2)
  - ##### Hidden layer 2: 32 neurons + ReLU + Dropout(0.2)
  - ##### Output: 1 neuron
- #### I decreased 20% of neuron powers and added early 
- #### stopping to avoid overfitting(model was free to have 40 epochs but stoped at 12) because validation score did not get better
- #### Best Thresholds = 0.5 , Because of detecting Fraudental transactions and Trade-off between FP,FN
- #### threshold scores for validation ==>  Precision: 89,Recall: 80
- #### Test data scores: Recall = 73, Precision = 85

# Model Selection :
- #### I selected MLP as best model because only scores are not matter ,Also perfomance when a lot of transactions are waiting to predic
- #### This is matter that model can handel it , Also MLP has more capacity to learn complex pattern and being tune in future 

## Report Guide
- Train/validation reports are saved in the `reports/` directory under each model's own folder
- Final test reports are saved in:
  - `reports/final_evaluate/knn/`
  - `reports/final_evaluate/tree/`
  - `reports/final_evaluate/logreg/`
  - `reports/final_evaluate/mlp/`