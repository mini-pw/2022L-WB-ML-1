install.packages("ISLR")

library(ISLR)
library(dplyr)
library(ggplot2)


View(Hitters)

Hitters_short <- Hitters[, c("Salary", "Hits", "Years")]

## Salary - wynagrodzenie
## Hits - liczba trafień, które wykonał w poprzednim roku
## Years - liczba lat, w których grał w lidze zasadniczej

summary(Hitters_short)

## braki danych dla zmiennej Salary - nasz target

Hitters_short <- Hitters_short %>%
  na.omit()

## rozkłady zmiennych

ggplot(Hitters_short, aes(x = Salary)) + 
  geom_density() #+
  #scale_x_log10()

library(rpart)
tree <- rpart(Salary~., data = Hitters_short)

library(rpart.plot)
rpart.plot(tree)
rpart.rules(tree)
rpart.predict(tree)
