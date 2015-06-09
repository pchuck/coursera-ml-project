---
title: "Coursera, Practical Machine Learning, Course Project"
author: patrick charles  
output:  
    html_document:
        keep_md: true
---

# Practical Machine Learning - Activity Type Analysis


## Summary

This [machine learning project](http://github.com/pchuck/coursera-ml-project/)
uses human activity sensor data to test the predictive capabilities of various
machine learning algorithms.

In particular, trees and random forests are used to build predictive models
using training data sets. Those models are then applied to test data sets
to see how effective they are at determining activity quality as an outcome.

The dataset being analyzed is provided by:
[Human Activity Recognition](http://groupware.les.inf.puc-rio.br/har)


## Prerequisites

The caret machine learning package is used for machine learning,
along with the rpart package for tree models. ggplot2 and rattle are
used for visualization.

```r
  if(!require(caret)) install.packages("caret", dep=T)
  if(!require(rpart)) install.packages("rpart", dep=T)
  if(!require(rattle)) install.packages("rattle", dep=T)
  if(!require(ggplot2)) install.packages("ggplot2", dep=T)
```


## Data


```r
  missingValues = c("NA","#DIV/0!", "") # recode missing values as type NA
  download.file(url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile="data/pml-training.csv", method="curl")
  pml.train.in <- read.csv("data/pml-training.csv", na.strings=missingValues)
```


```r
  download.file(url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile="data/pml-testing.csv", method="curl")
  pml.test.final <- read.csv("data/pml-testing.csv", na.strings=missingValues)
```


## Transformation

Columns which aren't relevant to the activity type, or contain NA values,
are removed from the training data set.


```r
  set.seed(2482)
  # remove columns containing only NA data
  removeTrain <- colSums(is.na(pml.train.in)) < nrow(pml.train.in)
  pml.train.clean <- pml.train.in[,removeTrain]
  removeTest <- colSums(is.na(pml.test.final)) < nrow(pml.test.final)
  pml.train.clean <- pml.train.in[,removeTest]
  # remove time stamps and other non-activity data
  pml.train.clean <- pml.train.clean[,-(1:5)]
```

The following variables remain after filtering out NA columns and
non-activity data.

```r
  names(pml.train.clean)
```

```
##  [1] "new_window"           "num_window"           "roll_belt"           
##  [4] "pitch_belt"           "yaw_belt"             "total_accel_belt"    
##  [7] "gyros_belt_x"         "gyros_belt_y"         "gyros_belt_z"        
## [10] "accel_belt_x"         "accel_belt_y"         "accel_belt_z"        
## [13] "magnet_belt_x"        "magnet_belt_y"        "magnet_belt_z"       
## [16] "roll_arm"             "pitch_arm"            "yaw_arm"             
## [19] "total_accel_arm"      "gyros_arm_x"          "gyros_arm_y"         
## [22] "gyros_arm_z"          "accel_arm_x"          "accel_arm_y"         
## [25] "accel_arm_z"          "magnet_arm_x"         "magnet_arm_y"        
## [28] "magnet_arm_z"         "roll_dumbbell"        "pitch_dumbbell"      
## [31] "yaw_dumbbell"         "total_accel_dumbbell" "gyros_dumbbell_x"    
## [34] "gyros_dumbbell_y"     "gyros_dumbbell_z"     "accel_dumbbell_x"    
## [37] "accel_dumbbell_y"     "accel_dumbbell_z"     "magnet_dumbbell_x"   
## [40] "magnet_dumbbell_y"    "magnet_dumbbell_z"    "roll_forearm"        
## [43] "pitch_forearm"        "yaw_forearm"          "total_accel_forearm" 
## [46] "gyros_forearm_x"      "gyros_forearm_y"      "gyros_forearm_z"     
## [49] "accel_forearm_x"      "accel_forearm_y"      "accel_forearm_z"     
## [52] "magnet_forearm_x"     "magnet_forearm_y"     "magnet_forearm_z"    
## [55] "classe"
```

'classe' (a measure of the quality of the activity) is the outcome
being predicted.


## Data Partitioning

The provided 'test' data set is further broken into two sets,
for training and model testing purposes, while the provided
final test data is set aside.


```r
  # separate training set into training and test subsets
  inTrain <-
    createDataPartition(y=pml.train.clean$classe, p=0.1, list=FALSE)
  pml.train <- pml.train.clean[inTrain, ]
  pml.test <- pml.train.clean[-inTrain, ]
```

The dimensions of the data sets after partitioning:

```r
  # The dimensions of the sets:
  dim(pml.train) # training
```

```
## [1] 1964   55
```

```r
  dim(pml.test) # testing
```

```
## [1] 17658    55
```

```r
  dim(pml.test.final) # 'hidden' or final test set
```

```
## [1]  20 160
```


## Exploratory Analysis

Some visualization of the data is performed to see the complexity of
the data and the relationship between some selected variables
(e.g. the pitch and yaw of the subject's forearm), compared to the
activity class outcome groupings.


```r
  ggplot(pml.train) + aes(x=pitch_forearm, y=yaw_forearm, color=classe) +
    xlab("Forearm Pitch") + ylab("Forearm Yaw") +
    geom_point(shape=19, size=2, alpha=0.3, aes(color=classe)) +
    ggtitle("Forearm Pitch and Yaw vs. Outcome")
```

![plot of chunk analysis.exploratory](figure/analysis.exploratory-1.png) 


## Machine Learning/Prediction - Tree Model

A tree model of type 'rpart' (recursive partitioning and regression tree)
is built using activity quality 'classe' as an outcome and all other
variables as predictors.

### Training


```r
  pml.tree <- train(classe ~ ., method="rpart", data=pml.train)
```

### Classification/Visualization

The structure of the classification tree and criteria can be visualized:

```r
  fancyRpartPlot(pml.tree$finalModel)
```

![plot of chunk tree.visualize](figure/tree.visualize-1.png) 

### Tree Model Accuracy / Predictions

The tree prediction model contains a summary of parameters and estimates of
accuracy.


```r
  pml.tree
```

```
## CART 
## 
## 1964 samples
##   54 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 1964, 1964, 1964, 1964, 1964, 1964, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa       Accuracy SD  Kappa SD  
##   0.03698435  0.5549200  0.43396300  0.03249661   0.04277309
##   0.03947368  0.5477336  0.42445049  0.02824959   0.03788272
##   0.12091038  0.3296200  0.07413283  0.04454737   0.06225282
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.03698435.
```

The estimated accuracy is **55.49%**

Let's see how that compares to actual accuracy on the training and test
sets. Activity quality outcomes can be predicted using the tree model and
compared with the actual classe variables in the training and test sets.


```r
  # percentage accuracy 
  perc = function(predicted, actual) {
    sum(predicted == actual) / length(actual) * 100
  }
  t.train.acc <- perc(predict(pml.tree, newdata=pml.train), pml.train$classe)
  t.test.acc <- perc(predict(pml.tree, newdata=pml.test), pml.test$classe)
```

Applying the tree model predictions on the training and test data sets,
and comparing the results with the actual activity quality outcome, yields:
* training set prediction accuracy: **56.01%**
* test set prediction accuracy: **56.60%**

**57%** accuracy isn't great, 
so let's try a more sophisticated model.


## Machine Learning/Prediction - Random Forest

### Model

Random forests use bootstrapping and many multiple tree generations to find an
optimal solution.


```r
  pml.rf <- train(classe ~., data = pml.train, method = "rf");
  pml.rf
```

```
## Random Forest 
## 
## 1964 samples
##   54 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 1964, 1964, 1964, 1964, 1964, 1964, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD  
##    2    0.9333335  0.9156118  0.009572021  0.01209635
##   28    0.9507401  0.9376595  0.008830032  0.01116250
##   54    0.9439861  0.9291123  0.009112155  0.01155805
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 28.
```

### Random Forest Accuracy / Predictions

Estimated accuracy is **95.07%**


```r
  rf.train.acc <- perc(predict(pml.rf, newdata=pml.train), pml.train$classe)
  rf.test.acc <- perc(predict(pml.rf, newdata=pml.test), pml.test$classe)
```

Applying the random forest predictions on the training and test data sets
and comparing the results with the actual activity quality outcomes yields:
* training set prediction accuracy: **100.00%**
* testing set prediction accuracy: **96.94%**

### Out of Sample Error and Cross Validation

Summary statistics for performance of the model on the test data
set, which was not used in the training of the random forest,
gives us an estimate of the sample error and a confidence interval.


```r
  pml.rf.test.predictions <- predict(pml.rf, pml.test)
  rfcm <- confusionMatrix(pml.test$classe, pml.rf.test.predictions)
  rfcm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4981   25    0    1   15
##          B   52 3262  103    0    0
##          C    0  100 2960   19    0
##          D    1    3   89 2791   10
##          E    0   23   46   54 3123
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9694          
##                  95% CI : (0.9667, 0.9719)
##     No Information Rate : 0.2851          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9612          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9895   0.9558   0.9256   0.9742   0.9921
## Specificity            0.9968   0.9891   0.9918   0.9930   0.9915
## Pos Pred Value         0.9918   0.9546   0.9614   0.9644   0.9621
## Neg Pred Value         0.9958   0.9894   0.9837   0.9950   0.9983
## Prevalence             0.2851   0.1933   0.1811   0.1622   0.1783
## Detection Rate         0.2821   0.1847   0.1676   0.1581   0.1769
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9931   0.9724   0.9587   0.9836   0.9918
```




```r
  pml.rf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 28
## 
##         OOB estimate of  error rate: 3.11%
## Confusion matrix:
##     A   B   C   D   E class.error
## A 555   1   1   0   1 0.005376344
## B  12 356   9   1   2 0.063157895
## C   0  12 330   1   0 0.037900875
## D   0   2   7 310   3 0.037267081
## E   0   3   1   5 352 0.024930748
```

* The accuracy of the random forest prediction model on the test set was
**96.94%**.

* The 95% confidence interval is **96.67%** to **97.19%**

* The estimated out of sample error rate is **~3.41%**


## Conclusion

Two machine learning algorithms were applied to predict activity 
quality outcomes from human activity sensor data.

Data was partitioned into training and test data sets and 
two algorithms, trees and random forest, compared in their 
efficacy for prediction of activity quality.

On the test data set
* The tree model achieved **56.60%** accuracy
* The random forest fared much better achieving **96.94%** accuracy

The random forest model generated has a 95% confidence interval of
**96.67%** to **97.19%** and estimated out of sample error
rate of **~3.41%**

Applied to the final 'hidden' test data set, the random forest model
successfully predicted 19 of 20 activity outcomes.




## Notes

The full analysis can be reproduced/generated using the following make target:
```
make render
```

The analysis can be viewed at:
  * [github/pchuck/coursera-ml-project/ml_project.md](https://github.com/pchuck/coursera-ml-project/blob/master/ml_project.md) (in markdown)
  * [pchuck.github.io/coursera-ml-project/ml_project.html](http://pchuck.github.io/coursera-ml-project/ml_project.html) (in html)
