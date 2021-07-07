import numpy as np
import matplotlib.pyplot as plt

def sigmoid(Z):
    """
    Sigmoid activation function. 
    """
    
    return 1/(1+np.exp(-Z))

def relu(Z):
    """
    ReLu activation function.
    """
    
    return np.maximum(0,Z)

def der_sigmoid(Z):
    """
    Derivative of sigmoid activation function.
    """
    
    return sigmoid(Z) * (1 - sigmoid(Z))

def der_relu(Z):
    """
    Derivative of ReLu activation function.
    """
    
    return np.sign(np.maximum(0, Z))

def layers(structure):
    """
    Function for initializing the weights and biases based on the NN structure.
    """
    
    #a dictionary where all of the weights and biases will be stored
    parameters = {}

    #iterating through the given layers
    for idx, layer in enumerate(structure, 1):
        
        #initializing weights and biases
        parameters['W'+str(idx)] = np.random.randn(
            layer["output"], layer["input"])*0.1 
        parameters['b'+str(idx)] = np.random.randn(
            layer["output"], 1)*0.1
        
    return parameters

def map_activation(Z, activation, typ):
    """
    Function for mapping the activation functions.
    """
    
    if activation=="relu":
        if typ == "backward":
            return der_relu(Z)
        else:
            return relu(Z)
    else:
        if typ == "backward":
            return der_sigmoid(Z)
        else:
            return sigmoid(Z)
    
def forw_prop(X, parameters, structure):
    """
    Function for forward propagation. 
    """
    
    #storing the values of A and Z
    #for the backpropagation step.
    storage = {}
    
    #initializing (i.e., first 'layer' is input)
    A_c = X
    
    #going through all layers given in the structure
    for idx, layer in enumerate(structure, 1):
        
        A_p = A_c #p: previous, c: current
        
        #W: weights
        #b: bias
        #Z: Wx+b
        #A: activation of Z
        W_c = parameters["W"+str(idx)]
        b_c = parameters["b"+str(idx)]
        Z_c = np.dot(W_c, A_p) + b_c
        A_c = map_activation(Z_c, layer["activation"], "forward")
        
        #adding to the storage
        storage["A"+str(idx-1)] = A_p
        storage["Z"+str(idx)] = Z_c
       
    #returning the current layer and the temporary storage
    return A_c, storage
    
def cost(y_hat, y):
    """
    Function for calculating the average binary cross-entropy.  
    """
    
    n = y_hat.shape[1]
    J = -1 / n * (np.dot(y, np.log(y_hat).T) + np.dot(1 - y, np.log(1 - y_hat).T))
    return np.squeeze(J)

def acc(y_hat, y):
    """
    Function for calculating the accuracy of the model.
    """
    
    y_prob = np.copy(y_hat)
    y_prob[y_prob>0.5] = 1        
    y_prob[y_prob<=0.5] = 0
    return np.mean((y_prob == y).all(axis=0))

def back_prop(y_hat, y, storage, parameters, structure, lr):
    """
    Function for backward propagation. 
    """
    
    y = y.reshape(y_hat.shape)
   
    dA_p = - (np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))
    
    #going through all layers in a reversed direction 
    for idx, layer in reversed(list(enumerate(structure))):
        
        dA_c = dA_p
        
        A_p = storage["A"+str(idx)]
        Z_c = storage["Z"+str(idx+1)]
        W_c = parameters["W"+str(idx+1)]
        b_c = parameters["b"+str(idx+1)]
   
        dZ_c = map_activation(Z_c, layer["activation"], "backward")*dA_c
        dW_c = np.dot(dZ_c, A_p.T) / A_p.shape[1]
        db_c = np.sum(dZ_c, axis=1, keepdims=True) / A_p.shape[1]
        dA_p = np.dot(W_c.T, dZ_c)
        
        #updating the weights and biases
        parameters["W"+str(idx+1)] -= lr * dW_c        
        parameters["b"+str(idx+1)] -= lr * db_c
    
    return parameters

def train(X, y, structure, epochs, lr=1e-3, visual=False):
    """
    Function for the training process.
    """
    
    #initializing the parameters
    parameters = layers(structure)
    
    #storing the accuracy and loss
    loss = []
    accuracy = []
    
    #iteratively training the model
    for idx in range(epochs):
        
        #forward propagation
        y_hat, storage = forw_prop(X, parameters, structure)
        
        #checking the loss and accuracy
        loss.append(cost(y_hat, y))
        accuracy.append(acc(y_hat, y))
        
        #backward propagation
        parameters = back_prop(y_hat, y, storage, parameters, structure, lr)
            
        #for the plot
        if visual:
            if idx%100==0:
                viz(idx, parameters)
    #returning the parameters, loss, and accuracy
    return parameters, loss, accuracy
