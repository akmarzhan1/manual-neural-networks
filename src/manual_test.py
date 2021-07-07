#importing the libraries
from sklearn.model_selection import train_test_split
from sklearn import datasets

N = 10000
#initializing the dataset
(X, y) = datasets.make_moons(n_samples=5000, noise=0.025, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#structure of the neural network
structure = [{"input": 2, "output": 25, "activation": "relu"},
             {"input": 25, "output": 50, "activation": "relu"},
             {"input": 50, "output": 50, "activation": "relu"},
             {"input": 50, "output": 25, "activation": "relu"},
             {"input": 25, "output": 1, "activation": "sigmoid"}]

#training the model N times
parameters, loss, accuracy = train(X_train.T, (y_train.reshape(((len(y_train), 1)))).T, structure, N, 0.01)

#plotting the loss
with plt.style.context('ggplot'):
    plt.plot(np.arange(0, N, 1), loss)
    
    
    
#testing accuracy and cross-entropy loss
y_hat, _ = forw_prop(np.transpose(X_test), parameters, structure)
test = acc(y_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
loss = cost(y_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
print('Testing Accuracy: {0} \nCross-Entropy Loss: {1}'.format(np.round(test*100, 3), np.round(loss, 3)))
