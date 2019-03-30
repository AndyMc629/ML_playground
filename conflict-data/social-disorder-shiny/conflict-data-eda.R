# data set exploration
data <- read.csv('data/events.csv')

library(tidyverse)
library(ggplot2)
library(dplyr)
ggplot(data=data) + geom_histogram(mapping = aes(x='COUNTRY'))

# Multiple line plot
ggplot(data, aes(x = BYEAR, y = )) + 
  geom_line(aes(color = COUNTRY), size = 1) +
  theme_minimal()

aggregate(x=data[, c("COUNTRY", "BYEAR")], by=list(mydf2$by1, mydf2$by2), FUN = mean)