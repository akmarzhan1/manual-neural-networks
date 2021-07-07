#importing the Neural Network packages
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#initializing the same NN
model = Sequential()
model.add(Dense(25, input_dim=2, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')


#checking the accuracy and loss
model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Testing Accuracy: {0} \nCross-Entropy Loss: {1}'.format(round(accuracy*100, 3), round(loss, 3)))
