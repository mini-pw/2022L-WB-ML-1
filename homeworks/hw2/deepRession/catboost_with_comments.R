library(dplyr)
library(naniar)
library(splitstackshape)
library(catboost)
set.seed(123)

'NIEPROFESJONALNY OPIS PARAMETRÓW I FUNCKJONALNOŚCI (NIE SPEŁNIAJĄCY STANDARDÓW
PACZKOWYCH)'

catboost_prepare_data <- function(data, target, continuous_names, drops, char_na){
  'This function prepares given data in order to preform catboost modeling on it.
   BRAK ZŁOŻONEGO OPISU CO ROBI KOD. TO JEST TYLKO CZARNA SKRZYNKA KTÓRA JAKOŚ
   TE DANE PRZYGOTUJE, ALE NIE WIADOMO JAK.
    
    
   BRAK MOŻLIWOŚCI WYBORU ROZMIARÓW ZBIORU TRENINGOWEGO I TESTOWEGO. DLA MAŁYCH
   ZBIORÓW BĘDZIE TAKA RELACJA JAK TU, ALE DLA ZBIORÓW RZĘDU KILKUDZIESIĘCIU 
   TYSIĘCY OBSERWACJI JUŻ MOŻNA CHCIEĆ DZIELIĆ PÓŁ NA PÓŁ
   It returns the list of data frames, which are:
    X_train - all independent variables from the sample of 70% of given data , 
              will be used to train the model
    X_test - the rest (30%) of independent variables, 
             which will be used to make predictions to test the accuracy of the model
    y_train - 70% of the records of dependent variable, corresponding to X_train 
    y_test - records of dependent variable, corresponding to X_test, 
            which will be used to test the accuracy between actual values and predictions
  
   Arguments:
    data - the data frame to prepare, BRAK OBSŁUGI INNYCH FORMATÓW (ALBO 
    INFORMACJI O TYM), NP. MATRIX
    target - targeted variable of the model BRAK INFORMACJI CZY TO MA BYĆ STRING
    DO NAZWY KOLUMNY, CZY CAŁA KOLUMNA
    continuous_names - a vector of names of continuous columns in data frame
    drops - a vector of names of highly correlated columns to drop,
            based on correlation matrix
    char_na - character or double used to substitute NA in the data frame'
  
  ## Data cleaning - removing data with nearly all variables blank 
  data <- data %>% mutate_all(funs(replace(., .== char_na, NA))) %>%
    na.omit()
  
  ## Encoding continuous variables by grouping
  'BARDZO SŁABY W SKUTKACH POMYSŁ. ZAŁÓŻMY ŻE MAMY POWIERZCHNIĘ DOMÓW JAKO ZMIENNĄ
  I MAMY W ZBIORZE GRUPĘ DOMÓW O POIWERCHNI 300M2 ALE WIĘKSZOŚĆ OBSERWACJI JEST
  Z ZAKRESU 40-100 M2. PREPROCESSING Z TAKIMI GRUPAMI WRZUCI JE WSZYSTKIE DO TEGO
  SAMEGO WORKA, A RÓŻNICA MIĘDZY 40 A 100M2 TO 2,5 KROTNOŚĆ ! A ICH CENY ZNACZNIE
  BĘDĄ SIĘ RÓŻNIĆ'
  
  data[continuous_names] <- data %>%select(continuous_names) %>% sapply(function(x){
    q <- quantile(x, probs = c(0.33,0.66,1))
    x <- cut(x, breaks = c(0,q[1],q[2],q[3]), 
             labels = c(1,2,3))
  })
  
  ## Encoding character data with integer labels
  'ZNOWU, SŁABY POMYSŁ, NADAJE WARTOŚCI ZMIENNYM KATEGORYCZNYM, PRZEZ CO DLA MODELU
  NASTĘPUJE ICH GRADACJA (JEDNA JEST LEPSZA OD DRUGIEJ)'
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
    params - list of parameters to base the model on, as for this version: NULL
  
  BRAK INFORMACJI O RETURNIE POZA WSTĘPEM, BRAK MOŻLIWOŚCI USTAWIENIA 
  HIPERPARAMETRÓW'
  
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
#######################  HUBERT  ####################### 
setwd("C:/Users/Hubert/Dropbox/Studia/Semestr VI/Warsztaty Badawcze/Review")
#######################  HUBERT  ####################### 
'BRAK WRZUCENIA DANYCH NA GITHUB PRZESZKADZA W TESTOWANIU'
bangalore <- read.csv('India/Bangalore.csv') # 'Price' will be the target variable
drops <- c("WashingMachine","BED","Microwave","TV","DiningTable","Sofa","Wardrobe",
           "Refrigerator","Wifi")
continuous_names <- c('Area','Price')
char_na <- 9

### Preparing the data
prepared_data <- catboost_prepare_data(bangalore,"Price",continuous_names,drops,char_na)
'ZOSTAWIENIE PRINTA W KODZIE'
#######################  HUBERT  ####################### 
View(bangalore)
View(prepared_data$X_train)
'ZAMIANA ZMIENNYCH KATEGORYCZNYCH NA LICZBY OD 1 DO X TO DUŻY BŁĄD, NADAJEMY MYLNĄ
INFORMACJĘ O HIERARCHII KTÓREJ MODEL SIĘ NAUCZY.

DODATKOWO W TYM KONKRETNYM PRZYPADKU LOCATION I AREA SĄ ŚCIŚLE POWIĄZANE (AREA MA 
TEN SAM NUMEREK DLA TYCH SAMYCH LOCATION) I TĄ INFORMACJĘ TRACIMY DALEJ

NIE ROZUMIEM ZAMIANY AREA NA CYFERKI O MNIEJSZYM ZAKRESIE.

'
#######################  HUBERT  ####################### 

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
'CZEMU SPROWADZAMY REGRESJĘ DO KLASYFIKACJI DLA ZMIENNEJ PRICE?'
#######################  HUBERT  #######################
'Accuracy'
sum(y_tested == y_test)/length(y_test)
#######################  HUBERT  ####################### 

#######################  HUBERT  ####################### 
lisbon <- read.csv('archive/lisbon-houses.csv') # 'Price' will be the target variable
drops <- c('Id','Country','District','Municipality')
continuous_names <- c('Bedrooms','AreaNet','AreaGross','Price.M2')
char_na <- 9

prepared_data <- catboost_prepare_data(lisbon,"Price",continuous_names,drops,char_na)

View(lisbon)
View(prepared_data$X_train)

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
library(Metrics)
mae(predict, y_test)

heart <- read.csv('archive/heart.csv') # 'Price' will be the target variable
drops <- NA
continuous_names <- c(1,2,3,4,6,7,8,9)
char_na <- 9

prepared_data <- catboost_prepare_data(heart,"chd",continuous_names,drops,char_na)

View(heart)
View(prepared_data$X_train)
'PRZY PREPROCESSINGU POWSTAŁY NA, MIMO ICH WCZEŚNIEJSZEGO BRAKU'
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
sum(y_tested == y_test)/length(y_test)

german <- read.csv('archive/german_credit_data_dataset.csv') # 'Price' will be the target variable
drops <- NA
continuous_names <- c('duration','age')
char_na <- 9

prepared_data <- catboost_prepare_data(german,"credit_amount",continuous_names,drops,char_na)

View(german)
View(prepared_data$X_train)
'PRZY PREPROCESSINGU POWSTAŁY NA, MIMO ICH WCZEŚNIEJSZEGO BRAKU'
### Assigning the variables
X_train <- prepared_data$X_train
X_test <- prepared_data$X_test
y_train <- prepared_data$y_train
y_test <- prepared_data$y_test

### Building the model
catboost_model <- catboost_function(X_train,y_train)

### Making the predictions
predict <- catboost_predict(catboost_model,X_test,y_test)
predict
mae(predict, y_test)
mean(predict)
'MAE STRASZNIE DUŻE, RZĘDU 930 DLA ŚREDNIEJ Z PREDICTA 2331. TO PRAWIE POŁOWA
ŚREDNIEJ WARTOŚCI'
#######################  HUBERT  ####################### 