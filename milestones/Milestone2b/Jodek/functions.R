library(lightgbm)

#' LightGBM_function - Train LightGBM model from a dataframe, with some automation.
#' 
#' @param data Data frame to perform prediction on. 
#' All columns must be <numeric>, <factor> (or <character> if `autofactor` parameter is set to `TRUE`).
#' @param target Target variable column name.
#' @param objective One of "regression", "binary", "multiclass".
#' @param autofactor Logical flag to control whether to encode <character> columns of `data` into <factor>. Default is `FALSE`. 
#' The encoding is likely to be dependent on order, which is not desirable.
#' @param ...  Additional model parameters to be passed to `lgb.train()`.
#'  Note that some parameters are set automatically by this wrapper function. These include
#'  * `...$objective` -> Controlled by `objective`
#'  * `...$num_class` -> Inferred from `data`.
#'  * `...$categorical_feature` -> Inferred from `data`.
#'  For list of possible parameters
#'  visit: https://lightgbm.readthedocs.io/en/latest/Parameters.html
#'
#' @return LightGBM model
#' @export
#'
#' @examples
#' # More examples present in `lgbm_test_*()` suite of functions
#' data(iris)
#' split1 <- sample(c(rep(0, 0.7 * nrow(iris)), rep(1, 0.3 * nrow(iris))))
#' train <- iris[split1 == 0, ]  
#' model <- lightGBM_function(
#'   train,
#'   "Sepal.Length",
#'   "regression"
#' )

lightGBM_function <- function(
  data,
  target,
  objective,
  ...,
  autofactor=FALSE,
  forceconvert=FALSE
  ){
  
  params <- list(...)
  stopifnot(
    # Support only <data.frome>
    "data.frame" %in% class(data),
    # Support only simple objectives
    (objective %in% c("regression", "binary", "multiclass")),
    is.null(params$objective),
    # Filled in by this function
    is.null(params$num_class),
    is.null(params$categorical_feature)
  )
  
  # Convert all <character> columns to <factor> if requested.
  # The encoding is likely to be dependent on order, which is not desirable.
  if (autofactor) {
    data <- lgbm_character_to_factor(data)
  } else {
    lgbm_assert_no_characters(data)
  }
  
  # Convert all <factor> columns to <numeric>.
  categorical_columns <- sapply(data, is.factor)
  data <- lgbm_factor_to_numeric(data)
  
  # Force conversion to <numeric> if requested
  if (forceconvert) {
    data <- data.frame(lapply(data, as.numeric))
  } else {
    lgbm_assert_all_numeric(data)
  }
  # Fill in parameters based on 
  if (objective == "regression") {
    params$objective = "regression"
  } else if (objective == "multiclass") {
    params$objective = "multiclass"
    params$num_classes = length(unique(data[, target]))
    # LGB expects a numeric vector starting from 0, R encodes factors from 1
    data[, target] <- data[, target] - 1L 
  } else {
    params$objective = "binary"
  }
  
  # Convert dataframe to lgb.dataset
  dataset <- lgbm_df_to_train(data, target, categorical_columns)

  # Train model
  model <- lgb.train(
    params = params,
    data = dataset
  )
  
  return(model)
}


##############################################

#' LightGBM_predict - predict target for samples in data.
#'
#' @param model Trained LightGBM model.
#' @param data Data frame to perform prediction on. 
#' All columns must be <numeric>, <factor> (or <character> if `autofactor` parameter is set to `TRUE`).
#' @param target Target column to drop for prediction or use in postprocessing (optional).
#' @param autofactor Logical flag to control whether to encode <character> columns of `data` into <factor>. Default is `FALSE`. 
#' The encoding is likely to be dependent on order, which is not desirable.
#' @param postprocess Logical flag to control whether to run postprocessing for the prediction. Default is `FALSE`.
#' This affects "multiclass" objective
#'
#' @return Vector of predicted values if `model$objective` is "binary", "regression", or "multiclass" and `postprocess` is `FALSE`
#' @return Matrix of shape `[n_rows, n_classes]` if `model$objective` is "multiclass" and `posprocess` is `TRUE`
#' @export
#'
#' @examples
#' # More examples present in `lgbm_test_*()` suite of functions
#' data(iris)
#' split1 <- sample(c(rep(0, 0.7 * nrow(iris)), rep(1, 0.3 * nrow(iris))))
#' train <- iris[split1 == 0, ]  
#' model <- lightGBM_function(
#'    train,
#'    "Sepal.Length",
#'    "regression"
#' )
#' test <- iris[split1== 1, ]
#' test <- test[,2:5]
#' predicted <- lightGBM_predict(model, test) 


lightGBM_predict <- function(
  model, 
  data, 
  target=NULL, 
  autofactor=FALSE, 
  postprocess=TRUE
  ){
  
  # Drop target if provided
  if (!is.null(target)) {
    data_target <- data[, target]
    data <- data[, !(names(data) %in% target)]
  }
  
  # Convert all <character> columns to <factor> if requested.
  # The encoding is likely to be dependent on order, which is not desirable.
  if (autofactor) {
    data <- lgbm_character_to_factor(data)
  } else {
    lgbm_assert_no_characters(data)
  }
  
  # Convert all <factor> columns to <numeric>.
  categorical_columns <- sapply(data, is.factor)
  data <- lgbm_factor_to_numeric(data)
  lgbm_assert_all_numeric(data)
  
  test <- as.matrix(data)
  
  prediction <- predict(model, test)
  
  if (postprocess && model$params$objective == "multiclass") {
    
    proba_matrix <- matrix(prediction, nrow = model$params$num_classes)
    proba_matrix <- t(proba_matrix)
    rownames(proba_matrix) <- rownames(data)
    
    if (is.null(target) || !is.factor(data_target)) {
      warning(
        "Postprocessing requested for multiclass objective, ",
        "but `target` not provided or `data$target` is not a factor. ",
        "Column names won't be set."
      )
    } 
    else {
      colnames(proba_matrix) <- levels(data_target)
    }
    
    return(proba_matrix)
  }
  
  return(prediction)
}

#################################################

#
# Example usage
#

#' 
#' Example regression for `Sepal.Length` from the `iris` dataset
#' 
#' @returns Mean squared error for the regression
#'
lgbm_test_regression <- function() {
  set.seed(123)
  
  data(iris)
  
  split1 <- sample(c(rep(0, 0.7 * nrow(iris)), rep(1, 0.3 * nrow(iris))))
  train <- iris[split1 == 0, ]  
  
  model <- lightGBM_function(
    train,
    "Sepal.Length",
    "regression"
  )
  
  test <- iris[split1== 1, ]
  test <- test[,2:5]
  
  predicted <- lightGBM_predict(model, test) 
  actual <- iris[split1 == 1, 1]
  mean((predicted - actual)^2)
}

#' 
#' Example binary classification for `Species` from the `iris` dataset
#' This function builds a model for predicting whether the `Species` is "setosa"
#' 
#' @returns Number of misclassified observations
#'

lgbm_test_binary <- function() {
  data(iris)
  
  setosa <- iris
  setosa$Is.Setosa = as.numeric(iris$Species == "setosa")
  setosa$Species <- NULL
  
  split1 <- sample(c(rep(0, 0.7 * nrow(setosa)), rep(1, 0.3 * nrow(setosa))))
  train <- setosa[split1 == 0, ]  
  
  model <- lightGBM_function(train, "Is.Setosa", "binary")
  
  test <- setosa[split1== 1, ]
  pred <- lightGBM_predict(model, test, target = "Is.Setosa")
  
  sum(round(pred) != test[, "Is.Setosa"])
}

#' 
#' Example multiclass classification for `Species` from the `iris` dataset
#' This function builds a model for predicting `Species`.
#' 
#' @returns Misclassified observation
#'

lgbm_test_multiclass <- function() {
  data(iris)
  
  split1 <- sample(c(rep(0, 0.7 * nrow(iris)), rep(1, 0.3 * nrow(iris))))
  train <- iris[split1 == 0, ]  
  model <- lightGBM_function(
    train,
    "Species",
    "multiclass"
  )
  
  test <- iris[split1== 1, ]
  pred <- lightGBM_predict(model, test, target = "Species", postprocess = TRUE)
  sum(apply(pred, 1, which.max) != as.numeric(test[, "Species"]))
}

################################################################################

#
# Helper functions
#

lgbm_assert_no_characters <- function(df) {
  if (any(sapply(df, is.character))) {
    stop(
      "Character columns present and `autofactor` is set to FALSE. ",
      "Character columns: ",
      paste0(colnames(df[,sapply(df, is.character)]), collapse=", ")
    )
  }
}
# Should fail
# lgbm_assert_no_characters(
#   data.frame(x = c("1","2"), y = c(1, 2), z = factor("a", "b"))
# )

lgbm_assert_all_numeric <- function(df) {
  if (!all(sapply(df, is.numeric))) {
    stop(
      "Non-numeric columns present after pre-processing. Problem columns: ",
      paste0(colnames(df[,! sapply(df, is.numeric)]), collapse=", ")
    )
  }
}
# Should fail
# lgbm_assert_all_numeric(
#   data.frame(x = c("1","2"), y = c(1, 2), z = factor("a", "b"))
# )

lgbm_character_to_factor <- function(df) {
  df[sapply(df, is.character)] <- lapply(
    df[sapply(df, is.character)],
    as.factor
  )
  df
}

lgbm_factor_to_numeric <- function(df) {
  df[sapply(df, is.factor)] <- lapply(
    df[sapply(df, is.factor)],
    as.numeric
  )
  df
}

lgbm_df_to_train <- function(df, target, categorical_columns) {
  target_values <- df[, target]
  
  features <- !(names(df) %in% target)
  data_features <- df[ , features]
  data_matrix <- as.matrix(data_features[, 1:ncol(data_features)])
  
  return(
    lgb.Dataset(
      data = data_matrix,
      label = target_values,
      free_raw_data = FALSE,
      categorical_feature = categorical_columns
    )
  )
}

