\name{XGBoost_function}
\alias{XGBoost_function}
\title{XGBoost_function}
\description{
Function which is presented below transform data (inplace) automatically to make it available to build model. Then function is building XGB model with default hyperparameters (if neither is given) or specified hyperparameters if any are given, fitting data to model and returning model. Then we can use this model to predict the data we are interested in.
}
\usage{
library(dplyr)
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
}
\arguments{
\item{data}{There we need to give data based on which we will   predict. It is well known as 'X' in machine learning literature.}

\item{target}{This parameter is this value which we want to predict based on 'data' parameter. It is well known as 'y' in machine learning literature.}

\item{booster}{(optional)
default value is gbtree. Another possibilities:
gblinear
dart}

\item{verbosity}{(optional)
Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug). Sometimes XGBoost tries to change configurations based on heuristics, which is displayed as warning message. If there’s unexpected behaviour, please try to increase value of verbosity.}

\item{validate_parameters}{(optional)
Default to false, except for Python, R and CLI interface. When set to True, XGBoost will perform validation of input parameters to check whether a parameter is used or not. The feature is still experimental. It’s expected to have some false positives.}

\item{nthread}{(optional)
Default to maximum number of threads available if not set. Number of parallel threads used to run XGBoost. When choosing it, please keep thread contention and hyperthreading in mind.}

\item{disable_default_eval_metric}{(optional)
Set automatically by XGBoost, no need to be set by user. Size of prediction buffer, normally set to number of training instances. The buffers are used to save the prediction results of last boosting step.}

\item{num_feature}{(optional)
Set automatically by XGBoost, no need to be set by user. Feature dimension used in boosting, set to maximum dimension of the feature

Parameters only for Tree Booster}

\item{eta}{(optional)
Default=0.3, alias: learning_rate. Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative. Range: (0, +inf)}

\item{gamma}{(optional)
Default=0, alias: min_split_loss. Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be. Range: (0, +inf).}

\item{max_depth}{(optional)
Default = 6. Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 is only accepted in lossguide growing policy when tree_method is set as hist or gpu_hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree. Range: (0, +inf) (0 is only accepted in lossguide growing policy when tree_method is set as hist or gpu_hist).}

\item{min_child_weight}{(optional)
Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be. Range:(0, +inf)}

\item{max_delta_step}{(optional)
Default = 0. Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update. Range: '(0, +inf)}
\item{subsample}{(optional)
Default = 1. Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration. Range: (0,1].}
\item{sampling_method}{(optional)
The method to use to sample the training instances.
Uniform: each training instance has an equal probability of being selected. Typically set subsample >= 0.5 for good results.
gradient_based: the selection probability for each training instance is proportional to the regularized absolute value of gradients (more specifically, sqrt(g^2 + lambda*h^2). Subsample may be set to as low as 0.1 without loss of model accuracy. Note that this sampling method is only supported when tree_method is set to gpu_hist; other tree methods only support uniform sampling.}
\item{max_leaves}{(optional)
Default=0. Maximum number of nodes to be added. Only relevant when grow_policy=lossguide is set.}
\item{refresh_leaf}{(optional)
This is a parameter of the refresh updater. When this flag is 1, tree leafs as well as tree nodes’ stats are updated. When it is 0, only node stats are updated.
More details of other parameters avaible there: https://xgboost.readthedocs.io/en/stable/parameter.html}
\item{objective}{(optional)
Default=reg:squarederror. Some of other possibilities: reg:squaredlogerror: regression with squared log loss, reg:logistic: logistic regression,
binary:logistic: logistic regression for binary classification, output probability}
\item{base_score}{(optional)
Default = 0.5. The initial prediction score of all instances, global bias. For sufficient number of iterations, changing this value will not have too much effect.}
\item{eval_metric}{(optional)
Evaluation metrics for validation data, a default metric will be assigned according to objective (rmse for regression, and logloss for classification, mean average precision for ranking). User can add multiple evaluation metrics. Some of possibilities: rmse: root mean square error, rmsle: root mean square log error,
mae: mean absolute error}
}
\value{
Function returns XGBoost model fitted to our data with specified hyperparameter if any were given to function.
Returned model can be used later to function predict to get predcitions we are interested in.
}
\references{
*https://xgboost.readthedocs.io/en/stable/R-package/xgboostPresentation.html
*https://cran.r-project.org/web/packages/xgboost/xgboost.pdf
*https://arxiv.org/pdf/1603.02754.pdf
*https://www.kaggle.com/code/rtatman/machine-learning-with-xgboost-in-r/notebook
*https://www.doit-intl.com/xgboost-or-tensorflow/
*https://towardsdatascience.com/xgboost-is-not-black-magic-56ca013144b4
*https://xgboost.readthedocs.io/en/stable/faq.html
*https://xgboost.readthedocs.io/en/stable/parameter.html
*http://www.chengli.io/tutorials/gradient_boosting.pdf
}
\author{
* Szymon Gut
* Wiktor Jakubowski
* Maria Kałuska
* Maciej Orsłowski
}
\note{
Function created for second milestone of Research Workshops subject on Warsaw University of Technology.
}
\seealso{
official XGB website: https://xgboost.readthedocs.io/en/stable/index.html.
Website of our Department (Faculty of Mathematics and Information Sciences): https://ww2.mini.pw.edu.pl
}