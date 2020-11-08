from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores_i = X[i].dot(W)
        correct_class_score = scores_i[y[i]]
        dW_Li  = np.zeros(W.shape)
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores_i[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW_Li[:,j] = X[i]
                dW_Li[:,y[i]] = dW_Li[:,y[i]] - X[i]
        
        dW += dW_Li 

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW = dW * (1/num_train) + 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_train = X.shape[0]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    Z = X.dot(W).transpose() # C x N matrix of all scores of all classes x samples 

    # broadcast subtract the true class scores and add the offset
    L = Z - Z[y, np.arange(num_train)] + 1
    
    # set losses of correct elements to zero so they are ignored
    L[y, np.arange(num_train)] = 0

    # set all negative loss elements to 0 
    L = L * (L > 0) 
    loss = (1 / float(num_train)) * np.sum(L) + reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #print(X.shape)
    #print(W.shape)
    #print(L.shape)
    # X is N x D
    # L is C x N
    # W is D x C

    # Set L to be the function 1(L > 0)
    L = L > 0
    L = L.astype(float)
    #print(L[:5,:5])
    # Calculate part of gradient due to incorrect-class terms
    dW_j = L.dot(X).transpose() # matrix of dimension D x C
    #print(dW_j.shape) 
    
    # calculate portion of gradient due to correct-score terms    
    #print(L.shape)
    l = L.sum(axis = 0)
    #print('l is shape {0}'.format(l.shape))
    X_L = X.transpose() * l
    X_L = X_L.transpose()
    #print(X_L.shape) # an N x D matrix

    # C x N matrix of 1s for every correct-class element
    Y_bool = np.zeros(L.shape)
    Y_bool[y, np.arange(num_train)] = 1
    #print(Y_bool.shape)

    # product of a C x N and N x D matrix is a C x D matrix 
    dW_yi = Y_bool.dot(X_L).transpose()

    # this issue could be the math having to do with calculating dW_j
    # and dW_yi separately and then subtracting them. Since 
    # dW = (1/N) * sum(dW_i, i) + reg * 2 * W
    #print(dW_yi.shape)
    dW = (1 / float(num_train)) * (dW_j - dW_yi) + reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
