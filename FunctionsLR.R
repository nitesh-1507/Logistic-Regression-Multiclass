# Function that implements multi-class logistic regression.
#############################################################
# Description of supplied parameters:
# X - n x p training data, 1st column should be 1s to account for intercept
# Y - a vector of size n of class labels, from 0 to K-1
# Xt - ntest x p testing data, 1st column should be 1s to account for intercept
# Yt - a vector of size ntest of test class labels, from 0 to K-1
# numIter - number of FIXED iterations of the algorithm, default value is 50
# eta - learning rate, default value is 0.1
# lambda - ridge parameter, default value is 0.1
# beta_init - (optional) initial starting values of beta for the algorithm, should be p x K matrix 

## Return output
##########################################################################
# beta - p x K matrix of estimated beta values after numIter iterations
# error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
# error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
# objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
LRMultiClass <- function(X, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = 0.1, beta_init = NULL){
  ## Check the supplied parameters
  ###################################
  # Check that the first column of X and Xt are 1s, if not - display appropriate message and stop execution.
  if(sum(X[, 1] == rep(1, nrow(X))) < nrow(X)){
    
    stop('The elements of first column in X are not all 1')
    
  }
  
  if(sum(Xt[, 1] == rep(1, nrow(Xt))) < nrow(Xt)){
    
    stop('The elements of first column in Xt are not all 1')
    
  }
  # Check for compatibility between X and Y
  if(nrow(X) != length(Y)){
    
    stop('The length of X and Y cannot be different')
    
  }
  
  # Check for compatibility between Xt and Yt
  if(nrow(Xt) != length(Yt)){
    
    stop('The length of Xt and Yt cannot be different ')
    
  }
  
  # Check for compatibility between X and Xt
  if(ncol(X) != ncol(Xt)){
    
    stop('The column size in X and Xt cannot be different')
    
  }
  
  # Check eta is positive
  if(eta <= 0){
    
    stop('Learning rate (eta) cannot be negative or zero')
    
  }
  
  # Check lambda is non-negative
  if(lambda < 0){
    
    stop('Regularization parameter (lambda) cannot be negative')
    
  }
  
  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes and construct corresponding pbeta. If not NULL, check for compatibility with what has been already supplied.
  if(length(beta_init) == 0){
    
    beta_init = matrix(0, ncol(X), length(unique(Y)))
    
  }else{
    
    if(nrow(beta_init) != ncol(X)){
      
      stop('The number of features in beta_init and X cannot be different')
    }
    
    if(ncol(beta_init) != length(unique(Y))){
      
      stop('The number of classes in beta_init and Y cannot be different')
    }
  }
    
    
  # sigmoid calculation
  sigmoid = function(x, beta){
      
      return(exp(x %*% beta))
    
    }
    
    
  # Vector to store training error
  error_train = rep(0, numIter+1)
    
  # Vector to store test error
  error_test = rep(0, numIter+1)
  
  # Vectore to store objective function value
  objective = rep(0, numIter+1)
  
  # Y actual
  Y_train = matrix(0, nrow(X), length(unique(Y)))
  
  # Assigning 1 if a class is present
  for(i in 1:ncol(beta_init)){
    
    Y_train[sort(unique(Y))[i] == Y, i] = 1
    
  }
    
  ## Calculate corresponding pk, objective value at the starting point, training error and testing error given the starting point
  ##########################################################################
 
  # Probability of each data point for each class
  prob_train = sigmoid(X, beta_init)
  prob_train = prob_train / rowSums(prob_train)
  
  # 1 - Probability of each data point for each class
  prob_train_0 = 1 - prob_train
  
  # Probability of each data point for each class
  prob_test = sigmoid(Xt, beta_init)
  prob_test = prob_test / rowSums(prob_test)
  
  # Class assignment train
  train_pred = apply(prob_train, 1, which.max) - 1
  
  # Class assignment test
  test_pred = apply(prob_test, 1, which.max) - 1
  
  # Error train set
  error_train[1] = sum(train_pred != Y) / length(Y) * 100
  
  # Error test set
  error_test[1] = sum(test_pred != Yt) / length(Yt) * 100
  
  # Objective function value
  objective[1] =  - sum(Y_train * log(prob_train, base = exp(1))) + (lambda/2) * sum(beta_init * beta_init)
  
  ## Newton's method cycle - implement the update EXACTLY numIter iterations
  ##########################################################################
 
  # Within one iteration: perform the update, calculate updated objective function and training/testing errors in %
  for(k in 1:numIter){
    
    # Matrix to store pk * 1-pk
    combined_prob = prob_train * prob_train_0
    
    # Update beta
    for(l in 1:ncol(beta_init)){
      
      # Calculates diag(W) %*% X
      W_X = X * combined_prob[, l]
      
      # Calculates t(X) %*% diag(W) %*% X
      product = crossprod(X, W_X)
      
      # Claculates (t(X) %*% diag(W) %*% X + lambda * Identity)
      inverse = solve(product + (lambda * diag(rep(1, ncol(X)))))
      
      # Update beta in each class
      beta_init[, l] = beta_init[, l] - eta * inverse %*% ((t(X) %*% (prob_train[, l]-Y_train[, l])) + lambda * beta_init[, l])
      
    }
    
    # Probability of each data point for each class in train set
    prob_train = sigmoid(X, beta_init)
    prob_train = prob_train / rowSums(prob_train)
    
    # 1 - Probability of each data point for each class in train set
    prob_train_0 = 1 - prob_train
    
    # Probability of each data point for each class in test set
    prob_test = sigmoid(Xt, beta_init)
    prob_test = prob_test / rowSums(prob_test)
    
    # class assignment train set
    train_pred = apply(prob_train, 1, which.max) - 1
    
    # class assignment test set
    test_pred = apply(prob_test, 1, which.max) - 1
    
    # Error train set
    error_train[k+1] = sum(train_pred != Y) / length(Y) * 100
    
    # Error test set
    error_test[k+1] = sum(test_pred != Yt) / length(Yt) * 100
    
    # Objective function value
    objective[k+1] =  - sum(Y_train * log(prob_train, base = exp(1))) + (lambda/2) * sum(beta_init * beta_init)
    
  }
  
  ## Return output
  ##########################################################################
  # beta - p x K matrix of estimated beta values after numIter iterations
  # error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
  # error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
  # objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
  return(list(beta = beta_init, error_train = error_train, error_test = error_test, objective =  objective))
}
