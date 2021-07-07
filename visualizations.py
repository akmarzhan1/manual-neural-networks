import os 
import matplotlib.pyplot as plt

#plotting function 
def plot(X, y, name, file_name=None, preds=None):
    
    #initializing the figure
    plt.figure(figsize=(16,12))
    plt.gca()
    plt.title(name, fontsize=20)

    #plotting 
    if preds is not None:
        b1, b2 = some_grid
        plt.contourf(b1, b2, preds.reshape(b1.shape), 25, alpha = 1, cmap=plt.cm.Spectral)
        plt.contour(b1, b2, preds.reshape(b1.shape), levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    if not file_name:
        plt.grid()
    plt.scatter(X[:, 0], X[:, 1], edgecolors='black', c=y.ravel(), cmap=plt.cm.Spectral)
    
    #saving the pictures
    if(file_name):
        plt.savefig(file_name)
        plt.close()
        
#plotting the original dataset
plot(X, y, "Original Data")


#initializing the grid 
some_grid = np.mgrid[-1.5:2.5:100j, -1:2:100j]

#visualizations for the manual NN
def viz(idx, parameters):
    title = "Manual NN: {:04}".format(idx)
    name = "manual_{:04}.png".format(idx//100)
    file = os.path.join('viz', name)
    pred_prob, _ = forw_prop(np.transpose(some_grid.reshape(2, -1).T), parameters, structure)
    pred_prob = pred_prob.reshape(pred_prob.shape[1], 1)
    plot(X_test, y_test, title, file_name=file, preds=pred_prob)
    
params_values = train(np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), structure, 8000, 0.01, visual=True)

import keras 

#visualizations for the Keras NN
def keras_viz(epoch, logs):
    title = "Keras NN - It: {:02}".format(epoch+1)
    name = "keras_nn_{:02}.png".format(epoch)
    file = os.path.join('viz', name)
    pred_prob = model.predict_proba(some_grid.reshape(2, -1).T, batch_size=32, verbose=0)
    plot(X_test, y_test, title, file_name=file, preds=pred_prob)

keras_visual = keras.callbacks.LambdaCallback(on_epoch_end=keras_viz)

#initialize the NN here
#I deleted this part so that it looks cleaner but it is the same code as in NN initialization
model.fit(X_train, y_train, epochs=45, verbose=0, callbacks=[keras_visual])
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Testing Accuracy: {0} \nCross-Entropy Loss: {1}'.format(round(accuracy*100, 3), round(loss, 3)))


#libraries
import glob
from PIL import Image

#paths
viz = "viz/*"
graph = "viz/graph_keras.gif"

#processing the images as GIF
img, *imgs = [Image.open(f) for f in sorted(glob.glob(viz))]
img.save(fp=graph, format='GIF', append_images=imgs, save_all=True, duration=80, loop=0)
