#
# This is the server logic of a Shiny web application. You can run the 
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)
library(dplyr)
library(ggplot2)

data <- read.csv('data/events.csv')

# Get the count of events for all countries in each year
country_year_group_count <- data %>%
  group_by(COUNTRY, BYEAR) %>%
  tally() 


# Define server logic required to draw series chart
shinyServer(function(input, output) {
   
  output$plot1 <- renderPlot({
    
    ggplot(country_year_group_count %>% filter(COUNTRY %in% input$country_name), aes(x = BYEAR, y = n)) + 
      geom_line(aes(color = COUNTRY), size = 1) +
      theme_minimal()
  })
  
})
