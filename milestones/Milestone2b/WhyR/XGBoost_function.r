
#' Function which is presented below transform data (inplace) automatically to make it available to build model. Then function is building XGB model with default hyperparameters (if neither is given) or specified hyperparameters if any are given, fitting data to model and returning model. Then we can use this model to predict the data we are interested in.
#'
#' @param data
#'There we need to give data based on which we will predict. It is well known as 'X' in machine learning literature.
#'
#' @param target
#'This parameter is this value which we want to predict based on 'data' parameter. It is well known as 'y' in machine learning literature.
#'
#' @param booster (optional)
#'default value is gbtree. Another possibilities:
#'gblinear
#'dart
#'
#' @param verbosity (optional)
#'Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug). Sometimes XGBoost tries to change configurations based on heuristics, which is displayed as warning message. If there’s unexpected behaviour, please try to increase value of verbosity.
#'
#' @param validate_parameters (optional)
#'Default to false, except for Python, R and CLI interface. When set to True, XGBoost will perform validation of input parameters to check whether a parameter is used or not. The feature is still experimental. It’s expected to have some false positives.
#'
#' @param nthread (optional)
#'Default to maximum number of threads available if not set. Number of parallel threads used to run XGBoost. When choosing it, please keep thread contention and hyperthreading in mind.
#'
#' @param disable_default_eval_metric (optional)
#'Set automatically by XGBoost, no need to be set by user. Size of prediction buffer, normally set to number of training instances. The buffers are used to save the prediction results of last boosting step.
#'
#' @param num_feature (optional)
#'Set automatically by XGBoost, no need to be set by user. Feature dimension used in boosting, set to maximum dimension of the feature
#'
#' Parameters only for Tree Booster
#' @param eta (optional)
#'Default=0.3, alias: learning_rate. Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative. Range: (0, +inf)
#' @param gamma (optional)
#'Default=0, alias: min_split_loss. Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be. Range: (0, +inf).
#'
#' @param max_depth (optional)
#'Default = 6. Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 is only accepted in lossguide growing policy when tree_method is set as hist or gpu_hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree. Range: (0, +inf) (0 is only accepted in lossguide growing policy when tree_method is set as hist or gpu_hist).
#'
#'@param min_child_weight (optional)
#'Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be. Range:(0, +inf)
#'
#' @param max_delta_step (optional)
#'Default = 0. Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update. Range: '(0, +inf)
#' @param subsample (optional)
#'Default = 1. Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration. Range: (0,1].
#'
#' @param sampling_method  (optional)
#'The method to use to sample the training instances.
#'Uniform: each training instance has an equal probability of being selected. Typically set subsample >= 0.5 for good results.
#'gradient_based: the selection probability for each training instance is proportional to the regularized absolute value of gradients (more specifically, sqrt(g^2 + lambda*h^2). Subsample may be set to as low as 0.1 without loss of model accuracy. Note that this sampling method is only supported when tree_method is set to gpu_hist; other tree methods only support uniform sampling.
#'
#' @param max_leaves (optional)
#'Default=0. Maximum number of nodes to be added. Only relevant when grow_policy=lossguide is set.
#'
#' @param refresh_leaf (optional)
#'This is a parameter of the refresh updater. When this flag is 1, tree leafs as well as tree nodes’ stats are updated. When it is 0, only node stats are updated.
#'
#'More details of other parameters avaible there: https://xgboost.readthedocs.io/en/stable/parameter.html
# Learning Task Parameters
#' @param objective (optional)
#'Default=reg:squarederror. Some of other possibilities: reg:squaredlogerror: regression with squared log loss, reg:logistic: logistic regression,
#'binary:logistic: logistic regression for binary classification, output probability
#'
#' @param base_score (optional)
#'Default = 0.5. The initial prediction score of all instances, global bias. For sufficient number of iterations, changing this value will not have too much effect.
#' @param eval_metric (optional)
#'  Evaluation metrics for validation data, a default metric will be assigned according to objective (rmse for regression, and logloss for classification, mean average precision for ranking). User can add multiple evaluation metrics. Some of possibilities: rmse: root mean square error, rmsle: root mean square log error,
#' mae: mean absolute error
#'
#' @return prediction using our model and given data.
#' @export
#'
library(dplyr)
library(xgboost)
library(data.table)
library(caTools)

### Function preprocessing data that it fits into xgboost model, returns splitted 
### dataset into train and test subsets
XGBoost_data_preprocessing <- function(data, target){
  # extracting labels
  labels <- data[, c(target)]
  
  # setting to data table format (recommended)
  data_DT <- data.table(data)
  
  # removing features with one unique value
  data_DT <- data_DT %>% select(where(~ n_distinct(.) > 1))
  
  # using one hot encoding on features
  data_enc <- model.matrix.lm(~.+0,data = data_DT[,-c(target),with=F], na.action = 'na.pass')
  data_enc
  # converting character / factor label to numeric
  if (is.character(labels)){
    labels <- as.numeric(factor(labels))-1
  }
  if (is.factor(labels)){
    labels <- as.numeric(labels)-1
  }
  
  # operations to split datasets into train and test
  set.seed(101)
  tmp <- as.data.frame(labels)
  colnames(tmp) <- c(target)
  tmp
  data_to_split <- as.data.frame(cbind(as.data.frame(data_enc), tmp))
  data_to_split
  sample <- sample.split(data_to_split[, c(target)], SplitRatio = 0.7)
  d_train <- subset(data_to_split, sample == TRUE)
  d_test  <- subset(data_to_split, sample == FALSE)
  setDT(d_train)
  setDT(d_test)
  
  # returning list of two data.table datasets: train and test
  data_output <- list("train" = d_train, "test" = d_test)
  return(data_output)
}

### Main function, building a model based on dataset (dataframe), 
### target variable name (string) and hyperparameters

XGBoost_function <- function(data, #dataset
                             target, #target variable (label)
                             nrounds = 50,
                             ...){
  
  # training set preprocessing
  d_train <- XGBoost_data_preprocessing(data, target)$train
  dtrain_matrix <- xgb.DMatrix(data = as.matrix(d_train[,-c(target),with=F]),
                               label = d_train[[target]], missing=NA)
  
  # setting parameters (if ommited, default values are used)
  params <- list(...)
  
  # returning a model
  xgb_model <- xgb.train(params = params, data = dtrain_matrix, nrounds=nrounds)
  
  return(xgb_model)
}

