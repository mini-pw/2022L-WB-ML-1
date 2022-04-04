library(dplyr)
library(naniar)
library(splitstackshape)
library(catboost)
set.seed(123)

catboost_prepare_data <- function(data, target, continuous_names, drops, char_na){
  'This function prepares given data in order to preform catboost modeling on it.
  
   It returns the list of data frames, which are:
    X_train - all independent variables from the sample of 70% of given data , 
              will be used to train the model
    X_test - the rest (30%) of independent variables, 
             which will be used to make predictions to test the accuracy of the model
    y_train - 70% of the records of dependent variable, corresponding to X_train 
    y_test - records of dependent variable, corresponding to X_test, 
            which will be used to test the accuracy between actual values and predictions
  
   Arguments:
    data - the data frame to prepare
    target - targeted variable of the model
    continuous_names - a vector of names of continuous columns in data frame
    drops - a vector of names of highly correlated columns to drop,
            based on correlation matrix
    char_na - character or double used to substitute NA in the data frame'
  
  ## Data cleaning - removing data with nearly all variables blank 
  data <- data %>% mutate_all(funs(replace(., .== char_na, NA))) %>%
    na.omit()
  
  ## Encoding continuous variables by grouping
  
  data[continuous_names] <- data %>%select(continuous_names) %>% sapply(function(x){
                                          q <- quantile(x, probs = c(0.33,0.66,1))
                                          x <- cut(x, breaks = c(0,q[1],q[2],q[3]), 
                                                   labels = c(1,2,3))
                                          })
  
  ## Encoding character data with integer labels
  data[, sapply(data, class) == 'character']<-  data %>% select(where(is.character)) %>%  
                                                sapply(function(x) as.integer(as.factor(x)) )
  
  ## Remove highly correlated variables
  data <- data[ , !(names(data) %in% drops)]
  
  ## Stratifying the data
  
  all <- stratified(data, c(target), 0.7, bothSets = TRUE)
  X_train <- all$SAMP1
  X_test <- all$SAMP2
  y_train <- X_train %>%
    pull(target) %>%
    as.numeric()
  print(y_train)
  y_test <- X_test %>%
    pull(target) %>%
    as.numeric()
  exclude <- c(target)
  X_train <- select(X_train, -exclude) %>% sapply(function(x){as.numeric(x)})
  X_test <- select(X_test, -exclude) %>% sapply(function(x){as.numeric(x)})

  list <- list("X_train" = X_train, "X_test" = X_test, "y_train" = y_train, "y_test" = y_test)
  
  return(list)
}

catboost_function <- function(X_train,y_train, params=NULL){
  'This function builds the catboost model with default parameters.
   The outcome of the function is mentioned model.
  
   Arguments:
    X_train - all independent variables from the sample of 70% of given data , 
              will be used to train the model, should be prepared by the function 
              `catboost_prepare_data`
    y_train - 70% of the records of dependent variable, corresponding to X_train,
              should be prepared by the function `catboost_prepare_data`
    params - list of parameters to base the model on, as for this version: NULL'
  
  ## Build the model
  
  train_pool <- catboost.load_pool(data = X_train, label = y_train)
  model <- catboost.train(train_pool, params = params)
  
  return(model)
}

catboost_predict <- function(model, X_test, y_test){
  'This function returns the predicitons of given data set, based on the used model
  
   Arguments:
    model - desired catboost model built with `catboost_function`
    X_test - the rest (30%) of independent variables, 
             which will be used to make predictions to test the accuracy of the model,
             should have been prepared with `catboost_prepare_data`
    y_test - y_test - records of dependent variable, corresponding to X_test, 
            which will be used to test the accuracy between actual values and predictions,
            should have been prepared with `catboost_prepare_data`'
  
  ## Predict the results from X_test
  
  test_pool <- catboost.load_pool(data = X_test, label = y_test)
  
  predict <- catboost.predict(model, 
                                 test_pool)
  
  return(predict)
}

## Example of use:

### Loading the data and necessary variables
bangalore <- read.csv('India/Bangalore.csv') # 'Price' will be the target variable
drops <- c("WashingMachine","BED","Microwave","TV","DiningTable","Sofa","Wardrobe",
           "Refrigerator","Wifi")
continuous_names <- c('Area','Price')
char_na <- 9

### Preparing the data
prepared_data <- catboost_prepare_data(bangalore,"Price",continuous_names,drops,char_na)

### Assigning the variables
X_train <- prepared_data$X_train
X_test <- prepared_data$X_test
y_train <- prepared_data$y_train
y_test <- prepared_data$y_test

### Building the model
catboost_model <- catboost_function(X_train,y_train)

### Making the predictions
predict <- catboost_predict(catboost_model,X_test,y_test)

### Rounding the predictions and compare the results
y_tested <- round(predict)
y_tested == y_test
