# problem Description:
This task is fraud detection, we gonna fit few models on kaggle fraud detection dataset with 284,315 samples
to predict a binary target (fraud or not)


# Dataset Overview:
#### We have 30 feature : time => bigscale, amount => big scale,v1 ... v28 =>small scale
#### time: times of transaction
#### amount: value of transaction
#### v1 ... v28: PCA transformed some features because they were private features
#### we should scale dataset because features scale are too different.
#### 0.017 % of observes are fraud transactions
#### there is no any missing value in rows

# Hypothesis before training:
## Question 1:
Given that ,we have a lot of samples and the problem is complex
I expect MLP will reach the best performance
on our data and in the second
place Decision Tree could find complex patterns
and solve the problem of fraud detection.
## Question 2:
In fraud detection recall is more important than Precision because we should not 
lose a fraud transaction as legitimate 
## Question 3:
Given that our dataset is imbalance if our model learn nothing and just classify every single observes as legitimate
The accuracy will be 99%, and This is why we wanna use another metrics such as F1 , Recall and Precision
## Question 4:
I expect feature scaling to significantly affect KNN performance because
KNN is a distance-based model.
If feature scales are different, 
features with larger scales will have a stronger effect on the distance values,
while features with smaller scales may be ignored by the model.
## Question 5:
Decision Tree is highly prone to overfitting because a tree could be too much complex 
and make a node for splitting every single observe, But we could control hyperparameters to avoid this thing

---
# After Training :
## Hyperparameters:
### KNN: 
for KNN algorithm we can tune number of neighbors and when model wanna decide
measure all distances and K smallest distances will be used for voting<br>
small k: risk of overfitting because model use a very small and sensitive to noise neighbor to vote<br>
large k: risk of underfitting because model use a very large set of neighbors to vote,therefore the model decision is too  generalized<br>
we will test different values for this hyperparameter to check where it will be balanced?
in this case it was 5 neighbor best value for K.

### Decision Tree:
we tuned max-depth for decision tree; depth in trees means how a tree can be deep
and split samples in more pure leaves
Too much max_depth: model will be very complex ,and it has risk of overfitting 
Too small max_depth : model will be very generalized , and will not learn the pattern 
in this case we choose 5 as max-depth
### logistic regression:
we tuned C for logistic regression : 1/C * penalty + loss is how C works <br>
if C be a large value penalty will be very small , the weights will be very large => overfitting<br>
if C be a small value penalty will be very large , the weight will be very small => under fitting we set this as 10
### MLP:
We trained an MLP with 30 input features, 64 neurons in the first hidden layer, 32 neurons in the second hidden layer, and 1 output neuron.
activation function between layers was reLU, last layer has linear out put because sigmoid has vanishing gradient risk 

## 1.Best score but slow model:
knn reached best score on validation data in training phase, FN = 11, Recall = 85
but knn has a problem , the KNN train and predict too slow !
because the model has to measure all distances
for each prediction ,
and best threshold for KNN = 0.3 because in this task FN error was more important for us
## * knn without scaling:
also we trained knn with unscaled datas and the performance was horrible
because: knn measure distances , if a feature has a bigger scale than other features, that feature will affect distance more than other features 
and feature important of this feature increases.
model fitted very bad and scores was too bad for more details => report\knn_not_scaled\

## 3.second place(same score with tree but faster than first model):
We trained an MLP with 30 input features, 64 neurons in the first hidden layer, 32 neurons in the second hidden layer, and 1 output neuron has a linear output because BCEwithlogistic loss applies sigmoid internaly by itself.
best threshold was = 0.3 scores was = 13 FN , Recall = 82% also MLP is so faster than KNN in predicting and this is so
matter in industry.
## 2.Third place:
we applayed grid search on our model to tune hyperparameters and this thing helped us to avoid overfitting for
Decision Tree , this model scores was great on validation datas 
FN = 13 in all thresholds and recall 82% in all thresholds it looks threshold has no effect on tree.
because: tree split samples in leaves , each leaf has probability = percentage of positives
and given that our tree worked nice on our data the probabilities was so near to 0 or 1 and threshold had no effect on it.
## worst model :
logistic regression was too simple to learn a complex task like this best threshold was = 0.3 
scores = 23 => FN , 69 => Recall 

# Choosen model:
I Choose MLP as best model , because MLPs has large capacity too learn and they are very faster than KNN,the scores of MLP is equal to KNN in the other hand,
also MLPs are highly tune able .

# How imbalance affect this task?
because of stratified cross-validation and tune models we controlled effect of imbalance datas as good as possible
and scores for MLP,Decision Tree , KNN was acceptable for fraud detection

# Running instruction
## installation:
- ### `pip install -r requirements.txt`<br>
## training:
- 1.run `src/data_prep.py`
- 2.run `src/train/run models one by one...`
- remember just run models like this:
* `src.train.model_name`
## FastAPI:
- 1.run in terminal `uvicorn src.api:app --reload`
- 2.Go to http://127.0.0.1:8000/docs
- 3.Copy your JSON and click on Execute

# Reflection Question :
## Question 1:
At first, we should talk about how accuracy measure : TP+TN / TOTAL <br>
how you see , if our model predict all samples as legitimate TN will be 99% and this could cover ,effect of low TP
So we should calculate metrics that measure given target class like f1, precision , recall.


# Question 2 Trade off FP&FN: 
In this mini-project FN is so matter , but it does not mean we should not pay attention to FP
it's too bad if a normal transaction classified as fraud , therefore despite I ranked models with Recall
I check F1 , Precision too , and models with threshold 0.3 had a good trade-off with FN and FP
reports are available in reports directory

# Question 3:
if I have an additional week to work on this project<br>
I will test more kind of neural networks with different architecture to achieve fancier results
Also in additional week I can fit a clustering algorith and use this kind of learning to check members behaviors.