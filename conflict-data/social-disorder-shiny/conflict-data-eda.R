# Get the first batch of data from the Urban Social Disorder Data set from "PRIO"
# https://www.prio.org/Data/Armed-Conflict/Urban-Social-Disorder/

# Aim is to perform some basic analyses on this and then show in a dashboard

# data set exploration
data <- read.csv('data/events.csv')

library(tidyverse)
library(ggplot2)
library(dplyr)
ggplot(data=data) + geom_histogram(mapping = aes(x='COUNTRY'))

# Get the countries with the most events per year since 2000
top_count_countries_2000_onwards <- data %>%
  filter(BYEAR>2000) %>%
  group_by(COUNTRY) %>%
  tally() %>%
  arrange(desc(n)) %>%
  slice(1:10)

# Get the count of events for all countries in each year
country_year_group_count <- data %>%
  group_by(COUNTRY, BYEAR) %>%
  tally() 

# Multiple line plot
# Filter the data to only be the top 10 countries by events since 2000.
ggplot(country_year_group_count %>% 
         filter(
           COUNTRY %in% top_count_countries_2000_onwards$COUNTRY
           ), aes(x = BYEAR, y = n)) + 
  geom_line(aes(color = COUNTRY), size = 1) +
  theme_minimal()


# Where is Syria?
country_year_group_count %>% filter(COUNTRY %in% c("SYRIA", "Syria"))
# Let's plot Syrain events
ggplot(country_year_group_count %>% filter(COUNTRY %in% c("SYRIA", "Syria")), aes(x = BYEAR, y = n)) + 
  geom_line(aes(color = COUNTRY), size = 1) +
  theme_minimal()
# Very strange that they are so low (especially compared to Iraq)? 
# Is this because of data gathering or labelling issues?


# Some nice histogram examples
if(!require(devtools)) install.packages("devtools")
devtools::install_github("kassambara/ggpubr")
install.packages("ggpubr")
library(ggpubr)
# Create a histogram of events in year coloured by country
gghistogram(country_year_group_count %>% 
              filter(
                COUNTRY %in% top_count_countries_2000_onwards$COUNTRY
              ), x = "n",
            add = "mean", rug = TRUE,
            color = "COUNTRY", fill = "COUNTRY")


