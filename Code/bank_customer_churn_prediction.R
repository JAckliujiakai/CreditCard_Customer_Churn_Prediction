#0 Import the data set
library(readr)
bank_data <- read.csv("C:/Users/JAck/Desktop/mqm/ds in business/team final case/bank.data.csv")

#1 Data Exploration
#1.1 Explore the Raw Data Set

#Brief review of bank_data
str(bank_data)
summary(bank_data)

#Check the unique values for each column
library(dplyr)
unique_counts <- sapply(bank_data, n_distinct)
unique_counts

#Check the missing value of teh data set
missing_counts <- colSums(is.na(bank_data))
missing_counts
# we have no null value in the data set

#1.2 Explore the Features
#We found our target is Exited column. so Y=1 means customers will churn, else not churn
#Explore the numerical features: CreditScore, Age, Tenure, NumOfProducts, Balance, EstimatedSalary
library(ggplot2)
bank_data$Exited <- as.factor(bank_data$Exited)
par(mfrow=c(2, 3))
ggplot(bank_data, aes(x = Exited, y = CreditScore,color = Exited)) +
  geom_boxplot() +
  labs(x = "Exited", y = "CreditScore")+
  scale_fill_manual(values = c("0" = "blue", "1" = "red"))

ggplot(bank_data, aes(x = Exited, y = Age)) +
  geom_boxplot() +
  labs(x = "Exited", y = "Age")

ggplot(bank_data, aes(x = Exited, y = Tenure)) +
  geom_boxplot() +
  labs(x = "Exited", y = "Tenure")

ggplot(bank_data, aes(x = Exited, y = NumOfProducts)) +
  geom_boxplot() +
  labs(x = "Exited", y = "NumOfProducts")

ggplot(bank_data, aes(x = Exited, y = Balance)) +
  geom_boxplot() +
  labs(x = "Exited", y = "Balance")

ggplot(bank_data, aes(x = Exited, y = EstimatedSalary)) +
  geom_boxplot() +
  labs(x = "Exited", y = "EstimatedSalary")

#Understand categorical data: Geography, Gender, HasCrCard, IsActiveMember
par(mfrow=c(2, 2))
ggplot(bank_data, aes(x = Exited, fill = Geography)) +
  geom_bar(position = "dodge") +
  labs(x = "Exited", y = "Count") +
  scale_fill_discrete(name = "Geography")

ggplot(bank_data, aes(x = Exited, fill = Gender)) +
  geom_bar(position = "dodge") +
  labs(x = "Exited", y = "Count") +
  scale_fill_discrete(name = "Gender")

ggplot(bank_data, aes(x = Exited, fill = factor(HasCrCard))) +
  geom_bar(position = "dodge") +
  labs(x = "Exited", y = "Count") +
  scale_fill_discrete(name = "HasCrCard")

ggplot(bank_data, aes(x = Exited, fill = factor(IsActiveMember))) +
  geom_bar(position = "dodge") +
  labs(x = "Exited", y = "Count") +
  scale_fill_discrete(name = "IsActiveMember")

#2 feature processing

#separating data into training data and testing data
install.packages('caret')
library(caret)
set.seed(1256)
trainIndex <- createDataPartition(bank_data$Exited, p = 0.75, 
                                  list = FALSE,
                                  times = 1)
train_data <- bank_data[trainIndex, ]
test_data <- bank_data[-trainIndex, ]


# drop useless features
to_drop <- c('RowNumber','CustomerId','Surname','Exited')
train_data_clean <- train_data %>%
  select(-one_of(to_drop))

test_data_clean <- test_data %>%
  select(-one_of(to_drop))

#one hot encoding
train_data_clean$Gender <- ifelse(train_data$Gender == 'Male',1,0)

train_data_clean$Geography.France <- ifelse(train_data_clean$Geography == 'France',1,0)
train_data_clean$Geography.Germany <- ifelse(train_data_clean$Geography == 'Germany',1,0)
train_data_clean$Geography <- NULL

test_data_clean$Gender <- ifelse(test_data$Gender == 'Male',1,0)

test_data_clean$Geography.France <- ifelse(test_data_clean$Geography == 'France',1,0)
test_data_clean$Geography.Germany <- ifelse(test_data_clean$Geography == 'Germany',1,0)
test_data_clean$Geography <- NULL

#data standardization: scale numerical data
scaled_train_data <- scale(train_data_clean[,c(1,3,4,5,6,9)])

#data set define
x_train <- cbind(scaled_train_data , train_data_clean[,c(2,7,8,10,11)])
y_train <- train_data$Exited

x_test <- test_data_clean
y_test <- test_data$Exited

#data cleaning done! cheers!

#3 Model Training and Result Evaluation
#3.1 Model Training 1

rf <- train(x=x_train,y=y_train, method = 'rf',ntree = 100, trControl = k_fold)
kknn <- train(x=x_train,y= y_train, method = 'kknn', trControl = k_fold)
lr <- train(x=x_train,y=y_train, method = 'glm',trControl = k_fold)

library(caret)
k_fold <- trainControl(method = "cv", 
                       number = 5,    
                       verboseIter = TRUE)

#random forest 0.4217687 
rf <- train(x=x_train,y=y_train, method = 'rf',ntree = 100, trControl = k_fold)
rf$results
predictions_rf <- predict(rf, newdata = x_test)
accuracy_rf <- mean(predictions_rf == y_test)
cat("Random Forest Accuracy:", accuracy_rf, "\n")

#kk nearest neighbors 0.7214886 
kknn <- train(x=x_train,y= y_train, method = 'kknn', trControl = k_fold)
kknn$results
predictions_kknn <- predict(kknn, newdata = x_test)
accuracy_kknn <- mean(predictions_kknn == y_test)
cat("K Nearest Neighbors Accuracy:", accuracy_kknn, "\n")

#logistic regression 0.2040816
lr <- train(x=x_train,y=y_train, method = 'glm',trControl = k_fold)
lr$results
predictions_lr <- predict(lr, newdata = x_test)
accuracy_lr <- mean(predictions_lr == y_test)
cat("Logistic Regression Accuracy:", accuracy_lr, "\n")

#we got the highest predicted accuracy is knn which is 0.7214886 . Compared to our training data,
#our model is a little over fitting

#3.2 Find Optimal Hyperparameters: KKNN
library(kknn)
param_grid <- expand.grid(kmax = c(5, 7, 9),distance = c(1,2), kernel = c("optimal", "rectangular"))
kknn_model <- train(
  x = x_train, 
  y = y_train, 
  method = "kknn", 
  tuneGrid = param_grid,
  trControl = k_fold)
print(kknn_model$bestTune)

param_grid <- expand.grid(kmax = 9 ,distance =1 , kernel = "rectangular")
best_kknn_model <- train(x_train,y_train,method = "kknn",tuneGrid = param_grid,trControl = k_fold)
best_kknn_model$results
predictions_kknn <- predict(best_kknn_model, newdata = x_test)
accuracy_kknn <- mean(predictions_kknn == y_test)
cat("Best KK Nearest Neighbors Accuracy:", accuracy_kknn, "\n")

#3.3 Model Evaluation - Confusion Matrix (Precision, Recall, Accuracy)
#TP: correctly labeled real churn
#Precision(PPV, positive predictive value): tp / (tp + fp); 
#Total number of true predictive churn divided by the total number of predictive churn; 
#High Precision means low fp, not many return users were predicted as churn users.
#Recall(sensitivity, hit rate, true positive rate): tp / (tp + fn) Predict most postive or churn user correctly. 
#High recall means low fn, not many churn users were predicted as return users.
cal_evaluation <- function(classifier, cm) {
  tn <- cm[1, 1]
  fp <- cm[1, 2]
  fn <- cm[2, 1]
  tp <- cm[2, 2]
  accuracy <- (tp + tn) / sum(cm)
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  
  cat(classifier, "\n")
  cat("Accuracy is:", accuracy, "\n")
  cat("Precision is:", precision, "\n")
  cat("Recall is:", recall, "\n")
  cat("\n")
}

draw_confusion_matrices <- function(confusion_matrices) {
  class_names <- c("Not", "Churn")
  for (cm in confusion_matrices) {
    classifier <- cm[[1]]
    cm_matrix <- cm[[2]]
    cal_evaluation(classifier, cm_matrix)
  }
}

confusion_matrix <- list(list("KK nearest neighbor", table(y_test, predict(best_kknn_model, x_test))))

draw_confusion_matrices(confusion_matrix)

library(gplots) 
confusion_matrix <- table(y_test, predict(best_kknn_model, x_test))
tn <- confusion_matrix[1, 1]
fp <- confusion_matrix[1, 2]
fn <- confusion_matrix[2, 1]
tp <- confusion_matrix[2, 2]

cat("True Neg: ", tn, "\n")
cat("False Pos: ", fp, "\n")
cat("False Neg: ", fn, "\n")
cat("True Pos: ", tp, "\n")

install.packages("gridExtra")
library(gridExtra)
library(grid)
grid.newpage()
grid.table(data.frame(
  " " = c("Actual Negative", "Actual Positive"),
  "Predicted Negative" = c(tn, fn),
  "Predicted Positive" = c(fp, tp)
))


#Accuracy: Accuracy represents the proportion of correctly predicted samples to the total number of samples. In your example, Accuracy is 0.7414966, which means the model correctly predicted approximately 74.15% of the samples.

#Precision: Precision represents the proportion of samples predicted as the positive class (Churn) that are actually part of the positive class. In your example, Precision is 0.270903, indicating that about 27.09% of the samples predicted as Churn are truly Churn.

#Recall: Recall represents the proportion of actual positive class samples that the model successfully predicted as the positive class. In your example, Recall is 0.1591356, indicating that the model successfully captured about 15.91% of the positive class samples.


#4 Feature Importance Discussion
# use random forest model to discuss the feature importance
library(ggplot2)
rf <- train(x_train,y_train, method = 'rf',ntree = 100, trControl = k_fold)
importance_scores <- varImp(rf)$importance
importance_df <- data.frame(Feature = rownames(importance_scores), Importance = importance_scores)
colnames(importance_df) <- c("Feature", "Importance")
importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]

ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  labs(title = "Feature Importance", x = "Feature", y = "Importance") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# the most important variable is Age
# then is Balance, Estimate Salary, CreditScore and Numofproducts.

#Business aspects: we can give coupon to elder customers to make them stay
