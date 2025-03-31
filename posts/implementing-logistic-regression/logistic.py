import torch

class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))
        return torch.matmul(X, self.w)

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        s = self.score(X)
        return (s >= 0).float()

class LogisticRegression(LinearModel):

    def sigmoid(self, s):
        """
        Compute the sigmoid function for each entry in the score vector s. 

        ARGUMENTS: 
            s, torch.Tensor: the score vector. s.size() = (n,)
        
        RETURNS: 
            probs, torch.Tensor: vector of probabilities. probs.size() = (n,)
        """
        return 1 / (1 + torch.exp(-s))

    def loss(self, X, y):
        """
        Compute the empirical risk using the logistic loss function.

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        """
        scores = self.score(X)
        probs = self.sigmoid(scores)
        loss = torch.mean(-y * torch.log(probs) - (1 - y) * torch.log(1 - probs))
        return loss

    def grad(self, X, y):
        """
        Compute the gradient of the logistic loss function with respect to the weights w.
        
        ARGUMENTS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        """
        scores = self.score(X)
        probs = self.sigmoid(scores)
        errors = probs - y
        grad = torch.matmul(errors.unsqueeze(0), X).squeeze(0) / X.size(0)
        return grad

class GradientDescentOptimizer:

    def __init__(self, model):
        self.model = model 
        self.w_prev = None
    
    def step(self, X, y, alpha, beta):
        """
        Perform one step of gradient descent with momentum.

        ARGUMENTS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p)
            y, torch.Tensor: the target vector. y.size() == (n,)
            alpha, float: the learning rate.
            beta, float: the momentum coefficient.
        """
        grad = self.model.grad(X, y)
        if self.w_prev is None:
            self.w_prev = torch.clone(self.model.w)
        momentum = beta * (self.model.w - self.w_prev)
        w_new = self.model.w - alpha * grad + momentum
        self.w_prev = torch.clone(self.model.w)
        self.model.w = w_new