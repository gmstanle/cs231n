from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):

        # calculate scores
        f = X[i].dot(W)
        # correction for numerical stability
        f -= np.max(f)
        
        # q is our "estimated probability" in the cross-entropy equation
        q = np.exp( f[y[i]] ) / np.sum( np.exp(f) )

        # add loss of ith sample
        loss += -np.log(q)

        # calculate grad
        dW_Li  = np.zeros(W.shape)
        for j in range(num_classes):
            q_j = np.exp( f[j] ) / np.sum( np.exp(f) ) 
            if j != y[i]:
                dW_Li[:,j] = X[i] * q_j
            else:
                dW_Li[:,j] = - X[i] * (1-q)

        # add the sample gradient to the total
        dW += dW_Li

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    
    # Add regularization to the loss and grad.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    F = X.dot(W) # N x C matrix of all scores of all classes x samples 
    # numerical staiblity correction
    F_corr = (F.transpose() - np.max(F, axis=1)).transpose()
    F_exp = np.exp(F_corr)

    q = F_exp[np.arange(num_train), y] / np.sum(F_exp, axis=1)

    loss = -np.log(np.sum(q) / num_train)
    loss += reg * np.sum(W * W)


    # Calculate grad
    Q = (F_exp.transpose() / np.sum(F_exp, axis=1)).transpose()
    Q[np.arange(num_train), y] = Q[np.arange(num_train), y] - 1
    dW = X.transpose().dot(Q)

    dW /= num_train
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
