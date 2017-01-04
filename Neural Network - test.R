

## This code is from a tutorial from R bloggers
# https://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/
# data dictionary http://www.clemson.edu/economics/faculty/wilson/R-tutorial/analyzing_data.html

library(neuralnet)
library(boot)
library(plyr)
library(MASS)


# this is a code with Neural Network tests

set.seed(500)
data <- Boston

apply(data,2,function(x) sum(is.na(x)))

# split the data randomly and estimate a linear model
index <- sample(1:nrow(data),round(0.75*nrow(data)))
train <- data[index,]
test <- data[-index,]
lm.fit <- glm(medv~., data=train)
summary(lm.fit)
pr.lm <- predict(lm.fit,test)
MSE.lm <- sum((pr.lm - test$medv)^2)/nrow(test)


# scale the data - it is important to scale the data when working with Neural Networks
maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)

# Note that scale returns a matrix that needs to be coerced into a data.frame.
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))


train_ <- scaled[index,]
test_ <- scaled[-index,]


# Usually, if at all necessary, one hidden layer is enough for a vast numbers of applications
# As far as the number of neurons is concerned, it should be between the input layer size and the output layer size,
# usually 2/3 of the input size. At least in my brief experience testing again and again is the best solution since there is
# no guarantee that any of these rules will fit your model best.

n <- names(train_)
f <- as.formula(paste("medv ~ ", paste(n[!n %in% "medv"], collapse = "+")))

nn_model <- neuralnet(f, data = train_, hidden = c(5, 3), linear.output = T)

plot(nn_model)


# Remember that the net will output a normalized prediction, so we need to scale it back in order to make a meaningful comparison (or just a simple prediction).
pr.nn <- compute(nn_model, test_[, 1:13])

pr.nn_ <- pr.nn$net.result * (max(data$medv) - min(data$medv)) + min(data$medv)
test.r <- (test_$medv) * (max(data$medv) - min(data$medv)) + min(data$medv)

MSE.nn <- sum((test.r - pr.nn_)^2) / nrow(test_)

print(paste(MSE.lm, MSE.nn))
# the MSE for the NN is almost twice as small as the MSE for the linear model


# Once again, be careful because this result depends on the train-test split performed above.
# Below, after the visual plot, we are going to perform a fast cross validation in order to be more confident about the results.

par(mfrow = c(1, 2))



# cross validation of linear model and NN


# cross validate linear model
set.seed(200)
lm.fit <- glm(medv~.,data=data)
cv.glm(data,lm.fit,K=10)$delta[1]


# cross validate NN
set.seed(450)
cv.error <- NULL
k <- 10

pbar <- create_progress_bar('text')
pbar$init(k)

for(i in 1:k){
  index <- sample(1:nrow(data),round(0.9*nrow(data)))
  train.cv <- scaled[index,]
  test.cv <- scaled[-index,]
  
  nn <- neuralnet(f,data=train.cv,hidden=c(5,2),linear.output=T)
  
  pr.nn <- compute(nn_model,test.cv[,1:13])
  pr.nn <- pr.nn$net.result*(max(data$medv)-min(data$medv))+min(data$medv)
  
  test.cv.r <- (test.cv$medv)*(max(data$medv)-min(data$medv))+min(data$medv)
  
  cv.error[i] <- sum((test.cv.r - pr.nn)^2)/nrow(test.cv)
  
  pbar$step()
}


mean(cv.error)      # the average NN error from cross-validation is more than twice as small as the error from the linear model
cv.error

# boxplot for the error
boxplot(cv.error,xlab='MSE CV',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for NN',horizontal=TRUE)




