path <- "C://Users//Abhijeet//Documents//GitHub//Kaggle//Titanic_Machine_Learning_from_Disaster//My Implementation"
if (getwd() != path) {setwd(path)}

# Step 1
## Fetching data from the training and testing datasets
inp_train <- read.csv(file="data/train.csv", header=TRUE, sep=",", quote="\"", 
                  stringsAsFactors=FALSE, na.strings=c("NA",""))
inp_test <- read.csv(file="data/test.csv", header=TRUE, sep=",", quote="\"", 
                  stringsAsFactors=FALSE, na.strings=c("NA",""))


# Step 1: Fetching data from the training and testing files

str(inp_train)

# Step 2: Pre-processing.
## Step 2.1: Cleaning the training dataset.

# Passenger Id has no meaning for this analysis, so we remove it from the 
# training dataset.
train <-  subset(inp_train, select= -c(PassengerId))
# There are some columns which are more meaningful if we convert them to factors.
train$Survived <- factor(train$Survived, levels = c("0", "1"), 
                         labels = c("Not_Survived", "Survived"))
train$Pclass <- factor(train$Pclass)
train$Sex <- factor(train$Sex)
train$Embarked <- factor(train$Embarked)

# Now that all the proper columns are converted to factors, we move on to handling
# the "NA" fields.

## Step 2.2: Handling NA's
summary(train)
sum(is.na(train$Cabin))
sum(complete.cases(subset(train, select=-Cabin)))

# Even though it's not evident in the summary, We observe from the data that most
# of the NAs are because of the _Cabin_ column.Hence, if we ignore this column, we
# still have 80% data left.

# We'll create a **Naive** version of our train set and observe the performance.

# We first try the Naive version. 
# We are ignoring these columns for now.
# TODO: analyze any use of the Cabin column.
train_naive <-  subset(train, select= -Cabin)

# we ignore all NAs in the train set.
# TODO: Predict the missing Age and Embarked columns.
train_naive <- train_naive[complete.cases(train_naive),]
# In the Naive version, we also ignore the "Name"  and "Ticket" columns.
train_naive <- subset(train_naive, select= -c(Name, Ticket))
head(train_naive)
#
# Step 3: Analyze Plots
## Survivality
#{r echo = FALSE, message=FALSE}
library(ggplot2)
library(gridExtra)

# histogram for Survived column.
ggplot(data=train_naive, aes(Survived)) + geom_histogram(binwidth=0.5) + 
ggtitle("Fig.1: Histogram of Survivality")
#

We observe from this plot that almost 60% didn't survive the disaster.
## Survivality by Age and Sex
#{r echo = FALSE, message=FALSE}
# Jitter plot for Age, Sex and Survived columns.
ggplot(data=train_naive, aes(x=Age, y=Sex, color=Survived)) + 
    geom_point(position="jitter") + 
    ggtitle("Fig.2: Comparision of Survivality by Age and Sex") + 
    xlab("Age (in years)")
#

We observe from this plot that higher percentage of females across all ages
survived whereas for males, there was higher survival rate for children but 
lower survival rate for adults.This confirms the "Women and Children First" 
policy used for the lifeboats.
## Survivality by no. of Siblings/Spouse and Age
#{r echo=FALSE}
ggplot(data=train_naive, aes(x=SibSp, y=Age, color=Survived)) + 
    geom_point(position="jitter") + 
    ggtitle("Fig.3: Comparision of Survivality by Age and Sex") +
    xlab("No. of Siblings") + ylab("Age (in years)")
#

We observe from the above plot that as the no. of siblings/spouse increase for ages<20,
survivality has decreased.
We plot a histogram to confirm the above observation.
#{r echo=FALSE, message=FALSE}
library(scales)
ggplot(data=train_naive, aes(x=SibSp, fill=Survived)) + 
    geom_bar(aes(y=(..count..)/sum(..count..))) + 
    scale_y_continuous(labels=percent) + 
    ggtitle("Fig.4: Percentage plot of no.of Siblings/Spouse and Survivality") + 
    xlab("Percentage of counts") + ylab("No. of Siblings")
#

The distribution of the number of siblings/spouse is not uniform to prove our theory.
we plot another histogram and this time we show the relative frequencies.
#{r echo=FALSE, message=FALSE}
ggplot(train_naive, aes(x=SibSp, fill=Survived)) + geom_bar(position="fill") + 
    ggtitle("Fig.5: Relative Frequencies of no.of Siblings/Spouse and Survivality") + 
    xlab("No. of Siblings") 
#

The above plot confirms that as the no. of siblings/spouse increase for ages <20, the 
survivality chances decreases.
We also observe that the survival rate of people with zero siblings/spouse is around 10% 
lower than the survival rate of people with 1 sibling. 
From Fig.3, we can infer that this is because most people with zero siblings/spouse are
ages>20 who (from Fig.2) had lower survivality rate.

## Survivality by Age Group
#{r}
train_naive$AgeGroup <- cut(train_naive$Age, breaks = c(0, 8, 13, 18, 60, Inf), labels = c("Child", "Teenager", "Young Adult", "Adult", "Elder"), right=FALSE)
ggplot(data=train_naive, aes(AgeGroup, fill=Survived)) + geom_histogram(binwidth=1)
#

# Step 4: Create Data Partitions
We divide the training set into 3 parts, in a 60:20:20 manner.
#{r message=FALSE}
# TODO: Use different splitting methods: k-fold, repeat k-fold, leave-one out.
library(caret)
train_naive <- train_naive[, -9]
set.seed(123) # setting seed for reproducibility
# We first divide the training set into two parts in 60:40 ratio.
inTrain <- createDataPartition(y=train_naive$Survived, p = 0.6, list=FALSE)
training.train <- train_naive[inTrain,]
rest_train <- train_naive[-inTrain,]

set.seed(123) # setting seed for reproducibility
# We now divide the rest of the training set into two parts in 50:50 ratio.
# This splits the original training set in a 60:20:20 ratio.
inTest <- createDataPartition(y=rest_train$Survived, p = 0.5, list=FALSE)
# 20% data for validation set to select best classifier
training.classifier_valid <- rest_train[inTest,] 
# 20% data for validation set to calculate out-of-sample errors
training.oos_valid <- rest_train[-inTest,] 
#
# Step 5: Select the best classifier
We select among different classifiers (Decision trees, Random Forests, 
                                       Naive Bayes, etc.) by training on training.train set and and predicting on 
training.classifier_valid.
#{r message=FALSE}
# TODO: Use interactions b/w variables
library(klaR)
# Decision Trees
set.seed(123)
tree_Model <- train(Survived ~ ., data=training.train, method="rpart")
pred_Tree <- predict(tree_Model, newdata=training.classifier_valid[,-1])
conf_Tree <- confusionMatrix(pred_Tree, training.classifier_valid$Survived)
# Decision Tree Plot
library(rattle)
fancyRpartPlot(tree_Model$finalModel)

# Random Forest
#set.seed(123)
#rfimp <- rfImpute(Survived ~., data=training.train)
set.seed(123)
forest_Model <- train(Survived ~ ., data=training.train, method="rf", rfimp)
pred_Forest <- predict(forest_Model, newdata=training.classifier_valid[,-1])
conf_Forest <- confusionMatrix(pred_Forest, training.classifier_valid$Survived)

# Naive Bayes
set.seed(123)
naiveB_Model <- NaiveBayes(Survived ~ ., data=training.train)
pred_naiveB <- predict(naiveB_Model, newdata=training.classifier_valid[,-1])
conf_naiveB <- confusionMatrix(pred_naiveB$class, training.classifier_valid$Survived)

#pred1 <- predict(mod1, testing)
#pred2 <- predict(logit_Model, testing)
#qplot(pred1, pred2, color=wage, data=testing)

# Ensemble: Fit a model that combines predictors
df_Ensemble <- data.frame(pTree=pred_Tree, pForest=pred_Forest, 
                          pNaive=pred_naiveB$class, 
                          Survived=training.classifier_valid$Survived)
set.seed(123)
ensemble_Model <- train(Survived ~., method="gbm", data=df_Ensemble)
pred_Ensemble <- predict(ensemble_Model, df_Ensemble[,-4])
conf_Ensemble <- confusionMatrix(pred_Ensemble, df_Ensemble$Survived)

# Boost
set.seed(123)
boost_Model <- train(Survived ~., method="gbm", data=training.train)
pred_Boost <- predict(boost_Model, newdata=training.classifier_valid[,-1])
conf_Boost <- confusionMatrix(pred_Boost, training.classifier_valid$Survived)
#
Now that all the predictions are complete, we compare them
#{r}
# Comparison of predictions from each model
compare_df <- data.frame(Accuracy = c(conf_Tree$overall[1], 
                                      conf_Forest$overall[1], 
                                      conf_naiveB$overall[1],
                                      conf_Ensemble$overall[1],
                                      conf_Boost$overall[1]),
                         row.names = c("rpart", "rf", "NaiveB", "Ensemble", "Boost"))
compare_df
#
We observe that both **Random Forest** and **Boost** gives us the highest Accuracy,
we see how the above model fares with the training.oos_valid dataset
#{r}
pred_Tree_oos <- predict(tree_Model, newdata=training.oos_valid[,-1])
conf_Tree_oos <- confusionMatrix(pred_Tree_oos, training.oos_valid$Survived)

pred_Forest_oos <- predict(forest_Model, newdata=training.oos_valid[,-1])
conf_Forest_oos <- confusionMatrix(pred_Forest_oos, training.oos_valid$Survived)

pred_naiveB_oos <- predict(naiveB_Model, newdata=training.oos_valid[,-1])
conf_naiveB_oos <- confusionMatrix(pred_naiveB_oos$class, training.oos_valid$Survived)

df_ensemble_oos <- data.frame(pTree=pred_Tree, pForest=pred_Forest, 
                              pNaive=pred_naiveB$class, 
                              Survived=training.oos_valid$Survived)
pred_Ensemble_oos <- predict(ensemble_Model, df_ensemble_oos[,-4])
conf_Ensemble_oos <- confusionMatrix(pred_Ensemble_oos, df_ensemble_oos$Survived)

pred_Boost_oos <- predict(boost_Model, newdata=training.oos_valid[,-1])
conf_Boost_oos <- confusionMatrix(pred_Boost_oos, training.oos_valid$Survived)

compare_df_oos <- data.frame(Accuracy = c(conf_Tree_oos$overall[1], 
                                          conf_Forest_oos$overall[1], 
                                          conf_naiveB_oos$overall[1],
                                          conf_Ensemble_oos$overall[1],
                                          conf_Boost_oos$overall[1]),
                             row.names = c("rpart", "rf", "NaiveB", "Ensemble", "Boost"))
compare_df_oos
#
We observe that **Random Forest** gives us the highest out-of-sample accuracy.
We select this classifier for predicting the test set.
# Step 6: Predict on Test set
## Step 6.1: Clean Test set and syncc columns with training set
#{r}
# remove unnecesary columns
test_naive <- subset(test, select = -c(Name, Ticket, Cabin))
test_naive$Pclass <- factor(test_naive$Pclass)
test_naive$Sex <- factor(test_naive$Sex)
test_naive$Embarked <- factor(test_naive$Embarked)
test_naive2 <- test_naive[complete.cases(test_naive),]
#
## Step 6.2: predict on the classifier selected above
#{r}
pred_Forest_test <- predict(forest_Model, newdata=test_naive2)

# converting factor to numeric(Not_Survived=0, Survived=1)
final_pred <- as.numeric(pred_Forest_test=="Survived")
df <- data.frame(PassengerId=test_naive2$PassengerId, Survived=final_pred)

# The rows which were dropped, we are predicting them as 0(Not_Survived)
for (i in test_naive$PassengerId) {
    if (!(i %in% test_naive2$PassengerId)){
        df <- rbind(df, data.frame(PassengerId=i, Survived=0))
    }
}     
#
# Step 7: Write results to csv and upload to Kaggle
#{r}
write.csv(df, file="naive_prediction_rf.csv", row.names=FALSE))
#
We find that the above model results in an accuracy of 0.73684 in Kaggle.
That's bad! we revisit some of the above steps and try to improve on this.
But first, just for fun, let's predict on the test set using some of the other 
classfiers.
#{r}
# Decision Tree
pred_Tree_test <- predict(tree_Model, newdata=test_naive2)

# converting factor to numeric(Not_Survived=0, Survived=1)
final_pred <- as.numeric(pred_Tree_test=="Survived")
df <- data.frame(PassengerId=test_naive2$PassengerId, Survived=final_pred)

# The rows which were dropped, we are predicting them as 0(Not_Survived)
for (i in test_naive$PassengerId) {
    if (!(i %in% test_naive2$PassengerId)){
        df <- rbind(df, data.frame(PassengerId=i, Survived=0))
    }
}
write.csv(df, file="naive_prediction_tree.csv", row.names=FALSE))
#
Decision Tree scored 0.74641

#{r}
# Naive Bayes
pred_naiveB_test <- predict(naiveB_Model, newdata=test_naive2)

# converting factor to numeric(Not_Survived=0, Survived=1)
final_pred <- as.numeric(pred_naiveB_test$class=="Survived")
df <- data.frame(PassengerId=test_naive2$PassengerId, Survived=final_pred)

# The rows which were dropped, we are predicting them as 0(Not_Survived)
for (i in test_naive$PassengerId) {
    if (!(i %in% test_naive2$PassengerId)){
        df <- rbind(df, data.frame(PassengerId=i, Survived=0))
    }
}
write.csv(df, file="naive_prediction_naiveB.csv", row.names=FALSE))
#
Naive Bayes scored 0.73684

#{r}
# Boost
df_Ensemble_test <- data.frame(pTree=pred_Tree_test, pForest=pred_Forest_test, 
                               pNaive=pred_naiveB_test$class)
pred_Ensemble_test <- predict(ensemble_Model, newdata=df_Ensemble_test)

# converting factor to numeric(Not_Survived=0, Survived=1)
final_pred <- as.numeric(pred_Ensemble_test=="Survived")
df <- data.frame(PassengerId=test_naive2$PassengerId, Survived=final_pred)

# The rows which were dropped, we are predicting them as 0(Not_Survived)
for (i in test_naive$PassengerId) {
    if (!(i %in% test_naive2$PassengerId)){
        df <- rbind(df, data.frame(PassengerId=i, Survived=0))
    }
}
write.csv(df, file="naive_prediction_ensemble.csv", row.names=FALSE))
#
Ensemble scored 0.73684

#{r}
# Boost
pred_Boost_test <- predict(boost_Model, newdata=test_naive2)

# converting factor to numeric(Not_Survived=0, Survived=1)
final_pred <- as.numeric(pred_Boost_test=="Survived")
df <- data.frame(PassengerId=test_naive2$PassengerId, Survived=final_pred)

# The rows which were dropped, we are predicting them as 0(Not_Survived)
for (i in test_naive$PassengerId) {
    if (!(i %in% test_naive2$PassengerId)){
        df <- rbind(df, data.frame(PassengerId=i, Survived=0))
    }
}
write.csv(df, file="naive_prediction_boost.csv", row.names=FALSE))
#
Boost scored 0.74641

#We observe from the above accuracy values that our Naive model wasn't that good.
#We also observe that the choice of model rarely made any big impact in the accuracy.
#One of the reason might be because we ignore the NA values of age and Embarked.
#We also have to decide which variables are necessary and whether we should 
#consider interaction between variables or not.If yes, then among which variables?


family_name <- gsub("[, ]+[A-Za-z].*", "", train$Name)
train$family <- family_name

dd <- train
sort_dd <- dd[with(dd, order(family, Pclass, Embarked, Ticket)),]

test_dd <- sort_dd

familyGroup <- numeric()
familyGroup[1] <- 1
familyCount <- 1
for (i in 2:nrow(test_dd)){
    #print(i)
   # print(test_dd$family[i])
    #print(test_dd$family[i-1])
    if ((test_dd$family[i] == test_dd$family[i - 1]) & 
            ((test_dd$Pclass[i] == test_dd$Pclass[i-1])) &
            ((test_dd$Embarked[i] == test_dd$Embarked[i-1])) &
            ((test_dd$SibSp[i] != 0) | (test_dd$Parch[i] !=0)) &
            ((test_dd$SibSp[i-1] != 0) | (test_dd$Parch[i-1] !=0))){
        familyGroup[i] <- familyCount
    }
    else{
        familyCount = familyCount + 1
        familyGroup[i] <- familyCount
    }
}

View(cbind(sort_dd[,-c(10, 13)], familyGroup))

test_dd[familyGroup %in% which(table(familyGroup) > 1),] # all family members