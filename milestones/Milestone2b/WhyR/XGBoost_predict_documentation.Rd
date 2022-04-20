\name{XGBoost_predict}
\alias{XGBoost_predict}
\title{XGBoost_predict}
\usage{
library(xgboost)
library(data.table)
library(caTools)
set.seed(101)
# dataset from https://www.kaggle.com/datasets/camnugent/california-housing-prices
source('XGBoost_function.R')
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
}
\arguments{
\item{data}{There we need to give data based on which we will   predict. It is well known as 'X' in machine learning literature.}
\item{target}{This parameter is this value which we want to predict based on 'data' parameter. It is well known as 'y' in machine learning literature.}
\item{model}{There we need to give object=model which we want use to predict values eg. XGBoost or Decision Tree}
}
\value{
prediction using our model and given data.
}
\description{
Function which is presented below returns the prediction based on given model and X-data. There is necessary to give data already transformed so before using this function please firstly use XGBoost_data_preprocessing(test, target) as a test set (preprocessed in order to work with algorithm).
}
