
#' Function which is presented below transform data (inplace) automatically to make it available to build model. Then function is building XGB model with default hyperparameters (if neither is given) or specified hyperparameters if any are given, fitting data to model and returning model. Then we can use this model to predict the data we are interested in.
#'
#' @param test
#'There we need to give data based on which we will   predict. It is well known as 'X' in machine learning literature.
#'Test data needs to be preprocessed with XGB_data_preprocessing() function prior to use in XGB_predict() function
#'
#' @param target
#'This parameter is this value which we want to predict based on 'data' parameter. It is well known as 'y' in machine learning literature.
#'
#' @param model There we need to give object=model which we want use to predict values eg. XGBoost or Decision Tree
#' @return prediction using our model and given data.
#' @export
#'

library(xgboost)
library(data.table)
library(caTools)
set.seed(101)
source('XGBoost_function.R')

### predicting function, returns values of predictions based on provided model and test set
### which can be obtained by XGBoost_data_preprocessing(...)$test (see example below)

XGBoost_predict <- function(test, target, model){
  #test set preprocessing
  test_matrix <- xgb.DMatrix(data = as.matrix(test[,-c(target),with=F]),
                             label = test[[target]], missing=NA)
  # predicting values
  pred <- predict(model, test_matrix, reshape=T)
  return(pred)
}


# ---------- EXAMPLE - California housing prices dataset ---------- #
# dataset from https://www.kaggle.com/datasets/camnugent/california-housing-prices

df <- read.csv('housing.csv')

# building model
modelHouse <- XGBoost_function(df, 'median_house_value', 
                               eta = 0.1, max_depth  = 7, nrounds=79)

# preprocessing test dataset
test <- XGBoost_data_preprocessing(df, 'median_house_value')$test

#prediction call (prices of real estates) on test dataset
XGBoost_predict(test, 'median_house_value', modelHouse)
#real values of test dataset
test[['median_house_value']]


#prediction call (prices of real estates) on train dataset
train <- XGBoost_data_preprocessing(df, 'median_house_value')$train
XGBoost_predict(train, 'median_house_value', modelHouse)
#real values
train[['median_house_value']]
