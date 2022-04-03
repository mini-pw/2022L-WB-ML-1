
#' Function which is presented below transform data (inplace) automatically to make it available to build model. Then function is building XGB model with default hyperparameters (if neither is given) or specified hyperparameters if any are given, fitting data to model and returning model. Then we can use this model to predict the data we are interested in.
#'
#' @param data
#'There we need to give data based on which we will   predict. It is well known as 'X' in machine learning literature.
#'
#' @param model There we need to give object=model which we want use to predict values eg. XGBoost or Decision Tree
#' @return prediction using our model and given data.
#' @export
#'
XGB_predict <- function(data, model) {
  pred <- predict(model, data)
  return(pred)
}
