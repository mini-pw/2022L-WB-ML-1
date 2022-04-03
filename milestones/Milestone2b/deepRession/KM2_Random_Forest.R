library(dplyr)
library(zoo)
library(tidyr)
library(naniar)
library(caret)
library(devtools)
library(ranger)
library(Metrics)

#' Automatically converts other types of data to the data.frame type. 
#'
#' @param data    data provided as list, matrix, data.table or data.frame
#' @param target  string, name of the target column
#'
#' @return        returns data.frame object
#' @export
#'
#' @examples

to_DataFrame <- function(data, target) {
  
  if (any(class(data) == "list") | any(class(data) == "matrix") | any(class(data) == "data.table")) {
    data <- as.data.frame(data)
  }
  
  if (!any(class(data) %in% c("data.frame", "matrix", "data.table", "list"))) {
    stop("Data type is invalid")
  }
  
  if (nrow(data) == 0 | ncol(data) == 1) {
    stop("Data frame is empty or too small to create model.")
  }
  
  
  #if (!is.null(target) &
  #    (!is.character(target) | !(target %in% colnames(data)))) {
  #  stop("Invalid 'target'")
  #}
  
  
  return(data)
}

#' Deletes rows with missing value in traget column and replaces missing values 
#' with average or mode function
#'
#' @param data          data provided as list, matrix, data.table or data.frame
#' @param target        string, name of the target column
#' @param missing_num   additional parameter, value which describes missing value
#' for 'numeric' observations (other than NA)
#' @param missing_cat   additional parameter, value which describes missing value
#' for 'character' observations (other than NA)
#'
#' @return              returns data.frame object
#' @export
#'
#' @examples
null_exterminator <- function(data, target, missing_num = FALSE, missing_cat = FALSE){
  
  if (target %in% colnames(data)) {
    if (any(is.na(data[target]))) {
      paste("missing values in column 'target': ",
            sum(is.na(data[target])) / nrow(data) * 100 , "%.")
      data <- data %>% drop_na(target)
    }
  }
  
  if (!any(data == missing_num) | missing_num == FALSE) {
    print("Not covered missing numerical values")
  } else if (any(data == missing_num)) {
    data <- data %>%
      replace_with_na_if(.predicate = is.numeric,
                         condition = ~ .x == missing_num)
  }
  
  if (!any(data == missing_cat) | missing_cat == FALSE) {
    print("Not covered missing nominal values")
  } else {
    data <- data %>%
      replace_with_na_if(.predicate = is.character,
                         condition = ~ .x == missing_cat)
  }
  
  data <- data %>% mutate_if(is.numeric , zoo::na.aggregate)
  
  calc_mode <- function(x) {
    distinct_values <- unique(x)
    distinct_tabulate <- tabulate(match(x, distinct_values))
    distinct_values[which.max(distinct_tabulate)]
  }
  
  data <- data %>%
    mutate(across(everything(), ~ replace_na(.x, calc_mode(.x))))
  
  return(data)
}


#' Scales and standardizes the columns provided as a vector of column indexes 
#' or column names.
#'
#' @param data      data.frame to scale
#' @param to_scale  vector of column indexes or column names to scale
#'
#' @return          data.frame with scaled columns
#' @export
#'
#' @examples
scale_continuous <- function(data, to_scale) {
  
  if (!is.na(to_scale)) {
    data[to_scale] <- lapply(data[to_scale], 
                             function(x) c(scale(x, center = FALSE)))
    return(data)
  } else{
    print("Can't scale the data without continuous values")
    return(data)
  }
  
}

#' Binarizes dataframe columns which are characters and have only two unique
#' values inside of themselves
#'
#' @param data      data.frame to binarize
#'
#' @return          data.frame with binarized proper columns
#' @export
#'
#' @examples
binarize_categorical <- function(data){
  
  for (i in colnames(data)) {
    
    if (typeof(data[[i]]) == 'character') {
      levels <- c()
      
      for (el in unique(data[i])) {
        levels <- c(levels, el)
      }
      
      if (length(levels) == 2) {
        for (j in 1:length(levels)) {
          data[[i]][data[[i]] == levels[j]] <- (j - 1)
        }
      }
    }
  }
  
  return(data)
}

#' Divides dataset into 4 subsets of two training sets and two
#' testing sets for x and y. The user can set train_set size
#'
#' @param data          data.frame to split
#' @param target        string, name of the target column
#' @param train_size    double / float, value from range [0,1] which sets the
#' size of the training set
#'
#' @return              list of: train_x, train_y, test_x, test_y data.frames
#' @export
#'
#' @examples
#' 
train_test_split <- function(data, target, train_size =0.6){
  
  smp_size  <- floor(0.75 * nrow(data))
  
  set.seed(123)
  train_ind <- sample(seq_len(nrow(data)), size = smp_size)
  
  train_x   <- data[train_ind,][!(names(data) %in% target)]
  test_x    <- data[-train_ind,][!(names(data) %in% target)]
  
  train_y   <- data[train_ind,][target]
  test_y    <- data[-train_ind,][target]
  
  return(list(
    'train_x' = train_x,
    'train_y' = train_y,
    'test_x' = test_x,
    'test_y' = test_y
  ))
}

#' Performs one hot encoding on given dataframe. OHE is performed on character 
#' columns with more than two unique categories only.
#'
#' @param data      data.frame to binarize 
#' @param to_scale  column names or numbers which were used for scaling
#' @param target    string providing target column name
#'
#' @return          data.frame with one hot encoded categorical columns
#' @export
#'
#' @examples
one_hot_encode <- function(data, to_scale, target){
  
  uniqueness <- lapply(data, unique)
  str        <- ""
  to_rm      <- c()
  
  for (el in names(uniqueness)) {
    if (length(uniqueness[[el]]) > 2 && is.character(uniqueness[[el]])) {
      to_rm <- c(to_rm, el)
      if (str != "") {
        str <- paste(str, "+", el, sep = "")
      } else{
        str <- el
      }
    }
  }
  
  if (str != "") {
    
    dummy  <- dummyVars(paste(" ~ ", str), data = data)
    OHE_df <- data.frame(predict(dummy, newdata = data))
    
    if (target %in% colnames(data)) {
      if (!is.na(to_scale)) {
        encoded_df <- cbind(data[c(names(to_scale), target)], OHE_df)
      } else{
        encoded_df <- cbind(data[c(target)], OHE_df)
      }
    }
    else {
      if (!is.na(to_scale)) {
        encoded_df <- cbind(data[to_scale], OHE_df)
      } else{
        encoded_df = OHE_df
      }
    }
    encoded_df <- encoded_df[!names(encoded_df) %in% to_rm]
    
    return(encoded_df)
  } else{
    print("No columns to encode")
    return(data)
  }
  
}

#' Performs all preprocessing methods on the given data. The preprocessing 
#' methods are: conversion to DataFrame, managing null values, scaling continuous
#' data, binarizing categorical columns with 2 categories, one hot encoding other
#' columns.
#'
#' @param data         data provided as list, matrix, data.table or data.frame
#' @param target       string, name of the target column
#' @param to_scale     vector of column indexes or column names to scale
#' @param missing_num  additional parameter, value which describes missing value
#' for 'numeric' observations (other than NA)
#' @param missing_cat  additional parameter, value which describes missing value
#' for 'character' observations (other than NA)
#'
#' @return             preprocessed dataset ready to split.
#' @export
#'
#' @examples
preprocessing <- function(data, target, to_scale = NA, missing_num = FALSE, 
                          missing_cat = FALSE){
  
  data <- to_DataFrame(data, target)
  data <- null_exterminator(data, target, missing_num, missing_cat)
  data <- scale_continuous(data,to_scale = to_scale)
  data <- binarize_categorical(data)
  data <- one_hot_encode(data,to_scale,target)
  return(data)
  
}

#' Build a random forest model with given preprocessed data. Use a ranger library and
#' let change its most important hyperparameter. 
#'
#' @param data             
#' data preprocessed with 'preprocessing' function
#' 
#' @param target           
#' string, name of the target column 
#'    
#' @param num.trees        
#' number of trees in model, default 500.
#' 
#' @param max.depth        
#'  Maximal tree depth. A value of NULL or 0 (the default)
#'  corresponds to unlimited depth, 1 to tree stumps (1 split per tree).
#'  
#' @param min.node.size   
#' Minimal node size. Default 1 for classification, 
#' 5 for regression, 3 for survival, and 10 for probability.
#' 
#' @param mtry             
#'  Number of variables to possibly split at in each node.
#'  Default is the (rounded down) square root of the number variables.
#'  Alternatively, a single argument function returning an integer,
#'  given the number of independent variables.
#'  
#' @param splitrule        
#'  Splitting rule. For classification and probability estimation
#'  "gini", "extratrees" or "hellinger" with default "gini". 
#'  For regression "variance", "extratrees", "maxstat" or "beta" with default "variance".
#'   
#' @param seed             
#'  Random seed. Default is NULL, which generates the seed from R.
#'  Set to 0 to ignore the R seed. 

#' @return  random forest model ready to predict target   
#' @export              
#'
#' @examples
RandomForest_function <- function(data, target, num.trees = 500, max.depth = NULL,
                                  min.node.size = NULL, mtry = NULL,
                                  splitrule = NULL, seed = NULL){
  
  model <- ranger(
    dependent.variable.name = target,
    data = data,
    num.trees = num.trees,
    max.depth = max.depth,
    min.node.size = min.node.size,
    mtry = mtry,
    splitrule = splitrule,
    seed = NULL
  )
  
  return(model)
}



#' Predicts target variable using given preprocessed attributes 
#' 
#' @param model       
#' result model from RandomForest_function
#' 
#' @param data   
#' data preprocessed with preprocessing function
#' 
#' @param target      
#' string, name of the target column    

#' @return  vector of predictions  
#' @export
#'
#' @examples
RandomForest_predict <- function(model, data, target){
  
  predict <- predict(model,data)
  return(predict$predictions)
}



# Przyk³adowe dzia³anie zaimplementowanych metod
#setwd("C:/Users/Hubert/Dropbox/Studia/Semestr VI/Warsztaty Badawcze")
data <- read.csv("./archive/lisbon-houses.csv")
df   <- to_DataFrame(data, 'Price')

nrow(df) 
nulls1  <- rep_len(c(1,1,NA,1,NA),nrow(df) )
nulls2  <- rep_len(c("Lisboa","Lisboa","Lisboa",NA,"Lisboa","Lisboa"),nrow(df) )
nulls3  <- rep_len(c(1,NA,1,1,1,1,1,1),nrow(df))

df_null <- df
df_null['Id']       <- nulls1
df_null['District'] <- nulls2
df_null['Price']    <- nulls3
df_null

df_notnull <- null_exterminator(df_null, 'Price')
df_notnull
nrow(df) 
nrow(df_notnull)

head(df)

scaled1   <- scale_continuous(df, c('Bedrooms','Bathrooms','AreaNet','AreaGross'))
scaled2   <- scale_continuous(df, c(5,6,7,8))
head(scaled1)
head(scaled2)

binarized <- binarize_categorical(scaled1)
head(binarized)

encoded   <- one_hot_encode(binarized, to_scale = c('Bedrooms','Bathrooms','AreaNet','AreaGross'),target = "Price")
names(encoded)


### TESTY preprocessingu
example1 <- read.csv("./archive/heart.csv")
prepared_example1 <- preprocessing(example1, "chd", to_scale = c(1,2,3,4,6,7,8,9))
#head(prepared_example1)


example2 <- read.csv("./archive/menshoes.csv")
#bierzemy tylko sensowne kolumny
example2 <- example2[c(3,17,22,25,31,32)]

prepared_example2 <- preprocessing(example2, "prices_amountmax")
names(prepared_example2)
head(prepared_example2)

example3 <- read.csv("./archive/german_credit_data_dataset.csv")
prepared_example3 <- preprocessing(example3, "customer_type", c(2, 5, 13, 16), -100000, "?")
head(prepared_example3)
dim(prepared_example3)
dim(example3)

example4 <- read.csv("./archive/allegro-api-transactions.csv") %>%  sample_frac(0.01)

prepared_example4 <-
  preprocessing(example4, "price", c(8, 10, 12), missing_num = -1)
head(prepared_example4)

#### lisbon-houses
data     <- read.csv("./archive/lisbon-houses.csv")
data     <- preprocessing(data, "Price", to_scale = c("Bedrooms","Bathrooms","AreaNet","AreaGross"))
data     <- train_test_split(data = data, target = "Price",  train_size = 0.7)
Xy_train <- cbind(data$train_x, data$train_y)
Xy_test  <- cbind(data$test_x, data$test_y)
X_train  <- data$train_x
X_test   <- data$test_x



rf <- RandomForest_function(
  data = Xy_train,
  target = "Price",
)

predictions_train <- RandomForest_predict(
  model = rf,
  data = X_train,
  target  = "Price"
)

predictions_test <- RandomForest_predict(
  model = rf,
  data = X_test,
  target  = "Price"
)

rmse(Xy_train$Price, predictions_train)
rmse(data$test_y$Price, predictions_test)
