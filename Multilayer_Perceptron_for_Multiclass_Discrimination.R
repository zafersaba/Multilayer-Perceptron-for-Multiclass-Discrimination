# read data into memory
X <-as.matrix(read.csv("hw03_images.csv", header=FALSE))
Y_truth <- as.matrix(read.csv("hw03_labels.csv", header=FALSE))

#implement safelog. It is used because normal log() function may have infinity values.
safelog <- function(x) {
  return (log(x + 1e-100))
}

# get number of samples and number of features
N <- length(Y_truth)
D <- max(Y_truth)

# one-of-K-encoding
y_truth <- matrix(0, N, D)
y_truth[cbind(1:N, Y_truth)] <- 1

# split into train and test
x_train <- X[1:(N/2),]
x_test <- X[((N/2)+1):1000,]
y_train <- y_truth[1:(N/2),]
y_test <- y_truth[((N/2)+1):1000,]

# define sigmoid function
sigmoid <- function(a) {
  return (1 / (1 + exp(-a)))
}

# initalize W and v
W <- as.matrix(read.csv("initial_W.csv",header=FALSE))
v <- as.matrix(read.csv("initial_V.csv", header=FALSE))

# define the softmax function
softmax <- function(a) {
  scores <- a
  scores <- exp(scores - matrix(apply(scores, MARGIN = 1, FUN = max), nrow = nrow(scores), ncol = ncol(scores), byrow = FALSE))
  scores <- scores / matrix(rowSums(scores), nrow(scores), ncol(scores), byrow = FALSE)
  return (scores)
}


# set learning parameters 
eta <- 0.0005
epsilon <- 1e-3
max_iteration <- 500
H <- 20

Z <- sigmoid(cbind(1, x_train) %*% W)
y_predicted <- sigmoid(cbind(1, Z) %*% v)
objective_values <- -sum(y_train * safelog(y_predicted) + (1 - y_train) * safelog(1 - y_predicted))

# learn W and v using batch learning
iteration <- 1
while (1) {

    # calculate hidden nodes
    #cbind is used to arrenge the dimensions of matrixes for matrix multiplications
    Z <- sigmoid(cbind(1, x_train) %*% W)
    # calculate output nodes
    y_predicted <- softmax(cbind(1,Z) %*% v)
  
    delta_v <- eta *  t(cbind(1, Z)) %*% (y_train - y_predicted) 
    delta_W <- eta * t(t((y_train - y_predicted) %*% (t(v)) * cbind(1,Z) *(1-cbind(1,Z)))  %*% cbind(1, x_train)) 
    
    #drop the last column so that the size is 785x20
    delta_W <- delta_W[,-1]
    
    v <- v + delta_v    
    W <- W + delta_W
  
  Z <- sigmoid(cbind(1, x_train) %*% W)
  y_predicted <- softmax(cbind(1,Z) %*% v)
  objective_values <- c(objective_values, -sum(y_train * safelog(y_predicted) + (1 - y_train) * safelog(1 - y_predicted)))
  
  #Stop when the error is less than the threshold(epsilon) or when we reach the max_iteration
  if (abs(objective_values[iteration + 1] - objective_values[iteration]) < epsilon | iteration >= max_iteration) {
    break
  }
  
  iteration <- iteration + 1
}

print(W)
print(v)

# plot objective function during iterations
plot(1:(iteration + 1), objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

#generete data with the same sizes as y_predicted_train
y_new <- as.matrix(read.csv("hw03_labels.csv", header=FALSE))
y_train_new <- y_new[1:(N/2),]

#generate predictions for training data
y_predicted_train <- apply(y_predicted, MARGIN = 1, FUN = which.max)

# calculate confusion matrix
train_confusion_matrix <- table(y_predicted_train, y_train_new)

#findout predictions for test
Z_test <- sigmoid(cbind(1, x_test) %*% W)
y_predicted_test <- softmax(cbind(1,Z_test) %*% v)

# calculate predicted values
y_predicted_test <- apply(y_predicted_test, MARGIN = 1, FUN = which.max)

#calculate confusion matrix
y_test_new <- y_new[((N/2)+1):N,]
test_confusion_matrix <- table(y_predicted_test, y_test_new)

print(train_confusion_matrix)
print(test_confusion_matrix)

