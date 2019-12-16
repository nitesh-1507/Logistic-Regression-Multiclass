# Load the letter data
# Training data
letter_train <- read.table("Data/letter-train.txt", header = F, colClasses = "numeric")
Y <- letter_train[, 1]
X <- as.matrix(letter_train[, -1])

# Testing data
letter_test <- read.table("Data/letter-test.txt", header = F, colClasses = "numeric")
Yt <- letter_test[, 1]
Xt <- as.matrix(letter_test[, -1])

# [ToDo] Make sure to add column for an intercept to X and Xt
X = cbind(rep(1, nrow(X)), X)
Xt = cbind(rep(1, nrow(X)), Xt)

# Source the LR function
source("FunctionsLR.R")

# [ToDo] Try the algorithm LRMultiClass with lambda = 1 and 50 iterations. Call the resulting object out, i.e. out <- LRMultiClass(...)
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL)

# The code below will draw pictures of objective function, as well as train/test error over the iterations
plot(out$objective, type = 'o')
plot(out$error_train, type = 'o')
plot(out$error_test, type = 'o')

# Feel free to modify the code above for different lambda/eta/numIter values to see how it affects the convergence as well as train/test errors
practice_1 = LRMultiClass(X, Y, Xt, Yt, numIter = 100, eta = 0.1, lambda = 1, beta_init = NULL)
plot(practice_1$objective, type = 'o')
plot(practice_1$error_train, type = 'o')
plot(practice_1$error_test, type = 'o')

practice_2 = LRMultiClass(X, Y, Xt, Yt, numIter = 100, eta = 0.1, lambda = 0.5, beta_init = NULL)
plot(practice_2$objective, type = 'o')
plot(practice_2$error_train, type = 'o')
plot(practice_2$error_test, type = 'o')

# [ToDo] Use microbenchmark to time your code with lambda=1 and 50 iterations. To save time, only apply microbenchmark 5 times.
library(microbenchmark)
microbenchmark(
  LRMultiClass(X, Y, Xt, Yt, numIter = 50, lambda = 1),
  times = 5
)

# [ToDo] Report the median time of your code from bicrobenchmark above in the comments below

# Median time:  4.987 (in sec)

