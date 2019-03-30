# From here; https://philippbroniecki.github.io/ML2017.io/day7.html
library(ISLR)
attach(Wage)
library(splines)

# Generalized Additive Models
# Can do this using lm
gam1 <- lm(wage ~ bs(year, 4) + bs(age, 5) + education, data = Wage)

# Easier to use gam package
# gam with natural splines for year and age
gam.m3 <- gam(wage ~ ns(year, 4) + ns(age, 5) + education, data = Wage)


# We can use the plot() function as well as plot.gam() to plot our results. 
# The difference between cubic splines and natural splines fit to age and 
# education is very small.
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 1, 1), oma = c(0, 0, 4, 0))
# plot function behaves similar to plot.gam
plot(gam.m3, se = TRUE, col = "blue")
title("Natural splines for age and education", outer = TRUE)

# Compare to cubic splines
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 1, 1), oma = c(0, 0, 4, 0))
plot.gam(gam1, se = TRUE, col = "red")
title("Cubic splines for age and education", outer = TRUE)

# Use anova to find best performing model
gam.m1 <- gam(wage ~ ns(age, 5) + education, data = Wage)
gam.m2 <- gam(wage ~ year + ns(age, 5) + education, data = Wage)
anova(gam.m1, gam.m2, gam.m3, test = "F")

# We can see that looking at the summary() function would not have been enough 
# to determine whether a natural spline on year is appropriate. 
# The coeffiecient is highly significant. However, the F-test reveals that the 
# model is not significantly better than the model without a natural spline for year.
summary(gam.m3)

# We can make predictions using predict() just as we did before.
preds <- predict(gam.m2, newdata = Wage)

# The gam() function also allows fitting logistic regression GAM with the 
# family = binomial argument.
gam.lr <- gam(I(wage > 250) ~ year + ns(age, df = 5) + education, family = binomial, data = Wage)
par(mfrow = c(1, 3))
plot(gam.lr, se = TRUE, col = "green")


# Looking at the education variable, we see that the first category has extremely 
# wide error bands. We check a cross-table of education against the indicator of 
# whether age > 250. We see that age > 250 is never true for lowest category.
table(education, I(wage > 250))

# Based on this we exclude the lowest education category from our model.
gam.lr.s <- gam(I(wage > 250) ~ year + ns(age, df = 5) + education, family = binomial, data = Wage, subset = (education != "1. < HS Grad"))
plot(gam.lr.s, se = TRUE, col = "green")

