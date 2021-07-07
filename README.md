# Manual Neural Networks
An OOP-based manual neural network model for simple 2d dataset classification.


****

Problem Description
===================

Machine learning using neural networks (NNs) is becoming more and more accessible as various tutorials and packages, such as Tensorflow, Keras, and PyTorch, come in, allowing us to create complex models in just a few lines of code. It takes a few hours of research to build NNs that perform exceptionally well, which is why the tools are so popular among computer scientists. However, not all people understand how exactly NNs work, treating them as a black-box model. This is why it is interesting for me to explore the exciting world of NNs and the underlying mechanisms that allow them to be so successful in all types of regression/classification tasks, specifically. In this assignment, I will build a neural net from scratch based on various algebra concepts and algorithms (e.g., forward and backward propagation) to classify a linearly non-separable 2d dataset. I will also compare the performance of a manual NN with that from Keras.

Solution Specification
======================

Optimization
------------

I explain all of the steps in the next part, but to summarize what I want to find:

1.  **Objective function:** The average binary cross-entropy loss function.

2.  **Optimization problem:** Minimizing the objective function.

3.  **Decision variables:** The weights and the biases in the model.

4.  **Constraints:** There are no explicit constraints on the system, other than the order in which the weights are updated (i.e., from the last layer all the way to the back) since the weights and biases can take any value. However, there are some edge cases. For example,

    1.  The cross-entropy score is built in a way that the value of log(0) is undefined, which means that if the specific value of <img src="https://render.githubusercontent.com/render/math?math=\hat y"> is exactly 0 or 1, the value of the loss at that point will be undefined. This is very rare as we get probabilities outside 0 and 1 almost all the time. One way to deal with it would be to add some small number (e.g., <img src="https://render.githubusercontent.com/render/math?math=\epsilon=1e-5"> to the expressions, so that they are defined, and the loss function is not too far from the actual value.

    2.  “Breaking symmetry problem,” which we will discuss further in the assignment.

5.  **Feasible Set**: As mentioned, there are no explicit constraints for my type of optimization problem (i.e., non-linear classification on a 2d dataset), which is why the weights and biases can be of any value. However, it only works because the points in our dataset don’t really have meaning (i.e., very simplistic). Depending on the type of problem we are trying to solve, there might be different constraints. For example, if we wanted to use the neural network for harder tasks such as part-of-speech tagging, then we would have to assign some natural constraints. For instance, there is an intrinsic structure to the words in the sentence, which is why we wouldn’t see a verb followed by a verb (unless separated by a comma), so we could let the NN know about these nuances.

    Additionally, although the cost function might be convex in <img src="https://render.githubusercontent.com/render/math?math=\hat y">, it might not be in most cases convex in parameters <img src="https://render.githubusercontent.com/render/math?math=\theta"> (weights and biases) as <img src="https://render.githubusercontent.com/render/math?math=\hat y"> itself is a function of a collection of weights and biases. We care about the convexity in the parameters because these are our decision variables. Although NNs are made of convex parts, the composition of convex functions is not convex, so deep neural networks are not convex (Cornell University, 2017). Even the most simplistic NNs are neither convex nor concave, and the deeper the NN goes, the less convex it becomes. This is why it typically hardly converges to global optimum points, so we will get local optima (i.e., and rarely, global). However, we have a very simplistic dataset, so we’ll see how well the model performs.

Context
-------

What is a neural network? How does it work? To build a model from scratch, we need to understand how NNs work first. As the name suggests, NNs draw inspiration from the actual neural system, where each node represents a neuron, and the whole model simulates layers of densely interconnected nodes, which allow the model to process, learn and analyze information. NNs can be seen as universal function approximators and they can learn to approximate any function with enough nodes. They usually include the following general parts:

1.  **input layer:** receives the input and passes it through the rest of the artificial neural network.

2.  **hidden layers:** apply weights and biases to the input (i.e., the input of each new layer is the output from the previous layer) and outputs them through an activation function.

3.  **output layer:** last layer, which produces the outputs of the model.

There are some fundamental variables we need to define for the layers:

1.  **<span>[</span>W<span>]</span> weights:** parameters that transform the input.

2.  **<span>[</span>b<span>]</span> bias:** constant added to the transformed input to affect the value out of an activation function.

3.  **<span>[</span>Z<span>]</span> weighted input + bias:** intermediary value, which is fed into the activation function. It is defined as WX+b.

4.  **<span>[</span>A<span>]</span> activation:** function that defines the output of a given input and determines whether a node will fire up or not. They introduce non-linear properties into the NN, since without them, the whole model would just be a combination of linear functions.

Let’s grab a simple two-layer example to see the predicted output for a given input. For convenience, we will use the sigmoid activation function.

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\hat y = \sigma(W_2\sigma(W_1 X+b_1)+b_2)"> 
 </p>

As we see, the output of the previous layer is then fed into the new layer as an input. The output of the second layer is then the predicted final value (i.e., for classification, we can just turn probabilities into classes). This process is called **forward propagation**, and it is defined as calculating the output by passing it through the layers. Now, to optimize the model, we need some way to find out how good our model’s predictions are. In this assignment, I am going to work with binary classification. I will optimize the **average binary cross-entropy score as my objective function**, which is similar to using logistic regression that usually optimizes the log loss for all observations (i.e., training) (“Cross entropy,” n.d.). It is defined as: <img src="https://render.githubusercontent.com/render/math?math=E(\hat y, y) = -\frac{1}{n}\sum^n_{i=1}(ylog\hat y + (1-y)log(1-\hat y))"> Here y is the true output, and <img src="https://render.githubusercontent.com/render/math?math=\hat y"> is the predicted output. E basically gives us an understanding of how far these two are. If the prediction is perfect, the value goes to 0, which is why our task is to minimize the average binary cross-entropy function score.

This is where the optimization part comes in, as we can tweak the weights and biases so that they give us better and better predictions after each iteration. We will do that through a technique called back-propagation. In short, we use gradient descent, and after each iteration, we calculate the loss function’s gradient with respect to NN weights and update the weights and biases accordingly.

We would calculate the loss function E and propagate the loss to all of the previous layers, starting from the last layer by changing the layers’ associated weights and biases. Let’s start from the weight update equations: <img src="https://render.githubusercontent.com/render/math?math=w=w-\alpha \frac{\partial E}{\partial w}"> Here <img src="https://render.githubusercontent.com/render/math?math=\alpha"> is the learning rate (i.e., the amount by which we update the weights and biases during training). To find <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial E}{\partial \hat y}"> we need to use the chain rule recursively. For convenience, I will write down the general equations and chain rule (i.e., in the code, it is all correctly linked).

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial E}{\partial w}=\frac{\partial E}{\partial \hat y}\frac{\partial \hat y}{\partial z}\frac{\partial z}{\partial w}"> 
 </p>

To find the derivatives, I used the derivation rules (i.e., addition, reciprocal) and standard formulas. Below is the derivative of the loss function with respect to the output.

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial E}{\partial \hat y}=\frac{1}{n}\frac{y}{\hat y}+\frac{1-y}{1-\hat y}=\frac{1}{n}\frac{-y+y\hat y+\hat y - y\hat y}{\hat y(1-\hat y)}=\frac{1}{n}\frac{\hat y - y}{\hat y (1-\hat y)}"> 
 </p>

Below <img src="https://render.githubusercontent.com/render/math?math=\hat y"> is the same thing as A that we defined in the beginning as it is simply the output of the last layer (i.e., after activation). Also, I am using a lower-case z for convenience. Since we have different activations, we will use a different term depending on whether we have ReLu or sigmoid. The derivative of the sigmoid activation function with respect to z looks like this:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial \hat y}{\partial z}=-\frac{1}{(1+e^{-z})^2}(-e^{-z})=\frac{e^{-z}}{(1+e^{-z})(1+e^{-z})}=\frac{1}{1+e^{-z}}\frac{1+e^{-z}-1}{1+e^{-z}}=\hat y(1-\hat y)"> 
 </p>


This is true because by the reciprocal rule, if we take  <img src="https://render.githubusercontent.com/render/math?math=1+e^{-z}">  as some s(z), then we know that <img src="https://render.githubusercontent.com/render/math?math=\frac{1}{s(z)}'=-\frac{s'(z)}{s(z)^2}"> . Alsom by the addition and chain rules <img src="https://render.githubusercontent.com/render/math?math=(1+e^{-z})'=0+-e^{-z}=-e^{-z}"> . For ReLu, things are a little easier. We will denote the activation as a for easier coordination. <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial a}{\partial z}"> is 0 when z<0 and 1 when z>0. 


The derivative at 0 is undefined, but the actual values rarely are 0, and even if that is the case, we can add some minimal value (e.g., Then, the last bit of information we need is: <img src="https://render.githubusercontent.com/render/math?math=\epsilon=1e-5">  to it so that it doesn’t break.

Then, the last bit of information we need is: <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial z}{\partial \hat w}=x"> 
x here is the input of that specific layer, but we should note that the subsequent layers take the output of the previous layers as input. So, in the end, we get:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial L}{\partial w}=\frac{1}{n}\frac{\hat y - y}{\hat y (1-\hat y)}\hat y(1-\hat y)x=\frac{1}{n}(\hat y -y)x"> 
 </p>

This works similarly for the rest of the layers with ReLu activation function, but considering that the output of the layer was defined by ReLu. Also, the update equations for the bias are almost the same but instead of finding <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial z}{\partial w}"> , we just find <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial z}{\partial b}"> (i.e., which is 1, but we should sum over all data points so that we get a single bias term for each weight).

Analysis
========

When initializing the weights, an important thing to notice is that we shouldn’t set all weights with the same values as the model will never learn (i.e., if unequal weights have to be developed). This is called the “breaking symmetry problem” and happens because when the error is propagated back, all weights will move identically and stay the same. Thus, we initialized the weights with some small random numbers. A similar thing is true for biases. We can also re-train the model several times with different initial weights to get better results. However, it is not required in our case, as our dataset is simple.

I used the logic from the previous part and also got inspired by the equations in <a href=https://towardsdatascience.com/https-medium-com-piotr-skalski92-deep-dive-into-deep-networks-math-17660bc376ba>this resource</a> and built a manual neural network without the use of packages. I also built a Keras NN to compare (see Appendix for code and the loss function plot).

I used a dataset from the `scikit` library called moons. The original dataset looked like this (i.e., I only added little noise to see how well it would classify clean data):


<div align="center">
    <img width="791" alt="Original Dataset" src="https://user-images.githubusercontent.com/47840436/124806407-78c54680-df7e-11eb-8ccd-7d06987f247f.png"></center>
</div>

I then used a simple NN architecture with 25 weights (nodes) in the the first layer, 50 nodes in both the second and the third layers, 25 node in the fourth layer and one node in the last layer (i.e., output). The results were good. The manual NN got 100% accuracy and an average cross-entropy loss of 0.002 after 10000 iterations, while the Keras NN had the same accuracy but no loss at all after 150 epochs (i.e., I am pretty sure it would have had 0 loss after around 40 epochs as we see from the GIF). I made some visualizations inspired by the code from <a href=https://github.com/SkalskiP/ILearnDeepLearning.py>here</a>. Click on the references below the figures to see GIFs.

<div align="center">
<img width="771" alt="Manual NN" src="https://user-images.githubusercontent.com/47840436/124806456-88dd2600-df7e-11eb-8e87-9ba9babe1d90.png">
    </div>
<div align="center">

See the visualization <a href="https://drive.google.com/file/d/1lXhQxjYnQzmGocWepYnBa37lLtrYJs6S/view?usp=sharing">here</a>.
</div>

<div align="center">

<img width="668" alt="Keras NN" src="https://user-images.githubusercontent.com/47840436/124806442-837fdb80-df7e-11eb-9b45-4ed5f2f8be9a.png">
</div>
<div align="center">

See the visualization <a href="https://drive.google.com/file/d/1d7on0yPe8qv3MWCbXs4oRnZxEXdMyBWy/view?usp=sharing">here</a>.
</div>

As we see from the diagrams above, both the manual NN and the Keras NN (i.e., with an equivalent architecture) performed well (i.e., partially because the dataset was super clear). We can see that the red part refers to one class and the blue part refers to another class. Both models were able to separate this non-linearly separable points. However, on more complex datasets, the manual NN might not be as successful due to the simplicity (e.g., even for this simple data, it took around 7000 iterations to fit). I ended up trying the manual NN on different datasets, but it takes much more time (i.e., iterations) and more complex NN architectures to get good results. As mentioned, the Keras library takes around 175 times fewer iterations to get the same accuracy (i.e., about 7000 vs. 40 epochs), which is why the Keras library is superior. Still, it was a fun experience to understand how a NN works on the inside.
