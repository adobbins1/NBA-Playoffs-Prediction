# This project was created by Austin Dobbins to predict NBA Playoff Teams using various statistics. 
# The stats use for this project come from the 2018-2019 season. 
# The two models created for this project are the Random Forest Classification Model and the
# Naive Bayes Classification Model. The Random Forest Model is able to predict Playoff teams with 
# 100% accuracy. The Naive Bayes model is able to predict Playoff teams with 83.3% accuracy. 

library(ggplot2)
library(ggplot2)
library(purrr)
library(tidyr)
library(dplyr)
library(MASS)
library(randomForest)
library(caret)
library(e1071)
library(glmnet)
library(plotly)
library(aod)
library(caret)

# EDA and Data Cleaning/Transformaiton

par(mfrow = c(4,7))
hist(nba$W)
hist(nba$L)
hist(nba$WINPer)
hist(nba$MIN)
hist(nba$PTS)
hist(nba$FGM)
hist(nba$FGA)
hist(nba$`FG%`)
hist(nba$`3PM`)
hist(nba$`3PA`)
hist(nba$`3P%`)
hist(nba$FTM)
hist(nba$FTA)
hist(nba$`FT%`)
hist(nba$OREB)
hist(nba$DREB)
hist(nba$REB)
hist(nba$AST)
hist(nba$TOV)
hist(nba$STL)
hist(nba$BLK)
hist(nba$BLKA)
hist(nba$PF)
hist(nba$PFD)
hist(nba$PlusMin)
hist(nba$`Off Eff`)
hist(nba$`Def Eff`)


nba <- transform(
  nba,
  Playoffs=as.factor(Playoffs)
  
)

sapply(nba, class)

# Random Forest

# Split Data Into Testing and Validation Sets 
set.seed(100)
train <- sample(nrow(nba), 0.6*nrow(nba), replace = FALSE)
trainset <- nba[train,]
validset <- nba[-train,]
summary(trainset)
summary(validset)

# Create the Model
model1 <- randomForest(formula = Playoffs ~ ., data = trainset, importance = TRUE)
model1

# Predicting the Training Set
predtrain <- predict(model1, trainset, type = "class")
# Checking the Classification Accuracy of Training Set
table(predtrain, trainset$Playoffs)

# Accuracy 
confusionMatrix(predtrain, trainset$Playoffs)

# Predicing the Validation Set
predvalid <- predict(model1, validset, type = "class")
# Checking the Classification Accuracy of Validation Set
mean(predvalid == validset$Playoffs)
table(predvalid, validset$Playoffs) 

# Accuracy
confusionMatrix(predvalid, validset$Playoffs)



# Naive Bayes

# Creating Naive Bayes Model 
nbModel <- naiveBayes(formula = Playoffs ~ ., data = trainset)
nbModel

# Predicting Training Set
predtrain3 <- predict(nbModel, trainset, type = 'class')
table(predtrain3, trainset$Playoffs)

# Accuracy 
confusionMatrix(predtrain3, trainset$Playoffs)

# Predicting Validation Set 
predvalid3 <- predict(nbModel, validset, type = 'class')
table(predvalid3, validset$Playoffs)

# Accuracy
confusionMatrix(predvalid3, validset$Playoffs)