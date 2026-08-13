# Hypothesis before training:
## Question 1:
Given that ,we have a lot of samples and the problem is complex
I expect MLP will reach the best performance
on our data and in the second
place Decision Tree could find complex patterns
and solve the problem of fraud detection.
## Question 2:
In fraud detection recall is more important than Precision because we should not 
lost a fraud transaction as legitimate 
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