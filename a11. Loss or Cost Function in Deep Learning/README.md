# Loss Functions and Their Use In Neural Networks

## Overview of loss functions and their implementations

![](https://miro.medium.com/max/1400/0*zMFuVqqKFLayW0U7)



Loss functions are one of the most important aspects of neural networks, as they (along with the optimization functions) are directly responsible for fitting the model to the given training data.

This article will dive into how loss functions are used in neural networks, different types of loss functions, writing custom loss functions in TensorFlow, and practical implementations of loss functions to process image and video training data — the primary data types used for computer vision, my topic of interest & focus.

# Background Information

First, a quick review of the fundamentals of neural networks and how they work.

![](https://miro.medium.com/max/1400/0*2k2pwxplR6fNdJNi.jpg)


![](https://miro.medium.com/max/1280/0*H1u8AftS6-qv8tXI.gif)



**Neural networks**  are a set of algorithms that are designed to recognize trends/relationships in a given set of training data. These algorithms are based on the way human neurons process information.

This equation represents how a neural network processes the input data at each layer and eventually produces a predicted output value.

![](https://miro.medium.com/max/476/1*_RlmGhVg6-KGL3RT8eXuWw.png)



To  **train**  — the process by which the model maps the relationship between the training data and the outputs — the neural network updates its hyperparameters, the weights,  _wT_, and biases,  _b,_ to satisfy the equation above.

Each training input is loaded into the neural network in a process called  **forward propagation**. Once the model has produced an output, this predicted output is compared against the given target output in a process called  **backpropagation**  — the hyperparameters of the model are then adjusted so that it now outputs a result closer to the target output.

This is where loss functions come in.

![](https://miro.medium.com/max/1280/0*XWWbqWIJttTSfmwF.png)


# Loss Functions Overview

A  **loss function**  is a function that  **compares**  the target and predicted output values; measures how well the neural network models the training data. When training, we aim to minimize this loss between the predicted and target outputs.

The  **hyperparameters**  are adjusted to minimize the average loss — we find the weights,  _wT_, and biases,  _b_, that minimize the value of  _J_  (average loss).

![](https://miro.medium.com/max/752/1*EBJivCIFmbfIhNue5gug4A.png)



We can think of this akin to residuals, in statistics, which measure the distance of the actual  _y_  values from the regression line (predicted values) — the goal being to minimize the net distance.

![](https://miro.medium.com/max/1400/0*xjbUMyJa5Rl4VLYs.jpg)



# How Loss Functions Are Implemented in TensorFlow

For this article, we will use Google’s  **TensorFlow**  library to implement different loss functions — easy to demonstrate how loss functions are used in models.

In TensorFlow, the loss function the neural network uses is specified as a parameter in model.compile() —the final method that trains the neural network.

    model.compile(loss='mse', optimizer='sgd')

The loss function can be inputed either as a String — as shown above — or as a function object — either imported from TensorFlow or written as custom loss functions, as we will discuss later.

    from tensorflow.keras.losses import mean_squared_error  
    model.compiile(loss=mean_squared_error, optimizer='sgd')

All loss functions in TensorFlow have a similar structure:

    def loss_function (y_true, y_pred):   
        return losses

It must be formatted this way because the model.compile() method expects only two input parameters for the loss attribute.

# Types of Loss Functions

In supervised learning, there are two main types of loss functions — these correlate to the 2 major types of neural networks: regression and classification loss functions

1.  Regression Loss Functions — used in regression neural networks; given an input value, the model predicts a corresponding output value (rather than pre-selected labels); Ex. Mean Squared Error, Mean Absolute Error
2.  Classification Loss Functions — used in classification neural networks; given an input, the neural network produces a vector of probabilities of the input belonging to various pre-set categories — can then select the category with the highest probability of belonging; Ex. Binary Cross-Entropy, Categorical Cross-Entropy

## Mean Squared Error (MSE)

One of the most popular loss functions, MSE finds the average of the squared differences between the target and the predicted outputs

![](https://miro.medium.com/max/752/1*kjfms6RCnHVMLRSq75AD0Q.png)



This function has numerous properties that make it especially suited for calculating loss. The difference is squared, which means it does not matter whether the predicted value is above or below the target value; however, values with a large error are penalized. MSE is also a convex function (as shown in the diagram above) with a clearly defined global minimum — this allows us to more easily utilize  **gradient descent optimization**  to set the weight values.

Here is a standard implementation in TensorFlow — built into the TensorFlow library as well.

    def mse (y_true, y_pred):   
        return tf.square (y_true - y_pred)

However, one disadvantage of this loss function is that it is very sensitive to outliers; if a predicted value is significantly greater than or less than its target value, this will significantly increase the loss.

![](https://miro.medium.com/max/1326/0*dutW2qhu_nrSv93s.jpg)



## Mean Absolute Error (MAE)

MAE finds the average of the absolute differences between the target and the predicted outputs.

![](https://miro.medium.com/max/752/1*78B4XwuVtBbacdTFIkjOXQ.png)



This loss function is used as an alternative to MSE in some cases. As mentioned previously, MSE is highly sensitive to outliers, which can dramatically affect the loss because the distance is squared. MAE is used in cases when the training data has a large number of outliers to mitigate this.

Here is a standard implementation in TensorFlow — built into the TensorFlow library as well.

    def mae (y_true, y_pred):   
        return tf.abs(y_true - y_pred)

It also has some disadvantages; as the average distance approaches 0, gradient descent optimization will not work, as the function's derivative at 0 is undefined (which will result in an error, as it is impossible to divide by 0).

Because of this, a loss function called a  **Huber Loss**  was developed, which has the advantages of both MSE and MAE.

![](https://miro.medium.com/max/1400/1*_cBTm_xd7ixdvcby7ZRcAw.png)



If the absolute difference between the actual and predicted value is less than or equal to a threshold value, 𝛿, then MSE is applied. Otherwise — if the error is sufficiently large — MAE is applied.

![](https://miro.medium.com/max/1052/1*_S2wqIIqrNNDY7aZM4Ef7w.png)



This is the TensorFlow implementation —this involves using a wrapper function to utilize the threshold variable, which we will discuss in a little bit.

    def huber_loss_with_threshold (t = 𝛿):   
        def huber_loss (y_true, y_pred):   
            error = y_true - y_pred   
            within_threshold = tf.abs(error) <= t  
            small_error = tf.square(error)  
            large_error = t * (tf.abs(error) - (0.5*t))  
            if within_threshold:   
                    return small_error  
            else:   
                    return large_error  
        return huber_loss

## Binary Cross-Entropy/Log Loss

This is the loss function used in binary classification models — where the model takes in an input and has to classify it into one of two pre-set categories.

![](https://miro.medium.com/max/1400/1*gbe2JiIk6Vsf32ulhhahCA.png)


Classification neural networks work by outputting a vector of probabilities — the probability that the given input fits into each of the pre-set categories; then selecting the category with the highest probability as the final output.

In binary classification, there are only two possible actual values of y — 0 or 1. Thus, to accurately determine loss between the actual and predicted values, it needs to compare the actual value (0 or 1) with the probability that the input aligns with that category (_p(i)_  = probability that the category is 1; 1 —  _p(i)_  = probability that the category is 0)

This is the TensorFlow implementation.

    def **log_loss** (y_true, y_pred):   
        y_pred = tf.clip_by_value(y_pred, le-7, 1 - le-7)  
        error = y_true * tf.log(y_pred + 1e-7) (1-y_true) * tf.log(1-y_pred + 1e-7)  
        return -error

## Categorical Cross-Entropy Loss

In cases where the number of classes is greater than two, we utilize categorical cross-entropy — this follows a very similar process to binary cross-entropy.

![](https://miro.medium.com/max/1136/1*HOCJtpCyQzWX8Xp3H8Ez2w.png)



Binary cross-entropy is a special case of categorical cross-entropy, where  _M_  = 2 — the number of categories is 2.

# Custom Loss Functions

As seen earlier, when writing neural networks, you can import loss functions as function objects from the tf.keras.losses module. This module contains the following built-in loss functions:

![](https://miro.medium.com/max/1400/1*ayR4LJx7MabyIjVVJc3c7A.png)



However, there may be cases where these traditional/main loss functions may not be sufficient. Some examples would be if there is too much noise in your training data (outliers, erroneous attribute values, etc.) — which cannot be compensated for with data preprocessing — or use in unsupervised learning (as we will discuss later). In these instances, you can write custom loss functions to suit your specific conditions.

    def **custom_loss_function** (y_true, y_pred):   
        return losses

Writing custom loss functions is very straightforward; the only requirements are that the loss function must take in only two parameters: y_pred (predicted output) and y_true (actual output).

Some examples of these are 3 custom loss functions, in the case of a variational auto-encoder (VAE) model, from  _Hands-On Image Generation with TensorFlow_  by Soon Yau Cheong.

    def vae_kl_loss(y_true, y_pred):  
        kl_loss =  - 0.5 * tf.reduce_mean(1 + vae.logvar -    tf.square(vae.mean) - tf.exp(vae.logvar))  
        return kl_lossdef vae_rc_loss(y_true, y_pred):  
        #rc_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)  
        rc_loss = tf.keras.losses.MSE(y_true, y_pred)  
        return rc_lossdef vae_loss(y_true, y_pred):  
        kl_loss = vae_kl_loss(y_true, y_pred)  
        rc_loss = vae_rc_loss(y_true, y_pred)  
        kl_weight_const = 0.01  
        return kl_weight_const*kl_loss + rc_loss

Depending on the math of your loss function, you may need to add additional parameters — such as the threshold value 𝛿 in the Huber Loss (above); to do this, you must include a wrapper function, as TF will not allow you to have more than 2 parameters in your loss function.

    def custom_loss_with_threshold (threshold = 1):   
        def custom_loss (y_true, y_pred):   
            pass #Implement loss function - can call the threshold variable  
        return custom_loss


[Source](https://towardsdatascience.com/loss-functions-and-their-use-in-neural-networks-a470e703f1e9)




# Why not Mean Squared Error(MSE) as a loss function for Logistic Regression? 🤔



![](https://miro.medium.com/max/875/1*Cy95jZ6ff8vdujww5XgKng.png)

> In this blog post, we mainly compare “**log loss**” vs  **“mean squared error”**  for logistic regression and show that why  **log loss**  is recommended for the same based on empirical and mathematical analysis.

Equations for both the loss functions are as follows:

**Log loss:**

![](https://miro.medium.com/max/875/1*zU7AU2zALgFKl1DGoxPLWQ.png)

**Mean Squared Loss:**

![](https://miro.medium.com/max/875/1*hCRKJJPz-0O04a_cH7N-JQ.png)

In the above two equations

y: actual label

ŷ: predicted value

n: number of classes

Let's say we have a dataset with 2 classes(n = 2) and the labels are represented as  **“0”**  and  **“1”.**

Now we compute the loss value when there is a complete mismatch between predicted values and actual labels and get to see how log-loss is better than MSE.

**For example:**

Let’s say

-   Actual label for a given sample in a dataset is “1”
-   Prediction from the model after applying sigmoid function = 0

**Loss value when using MSE:**

(1- 0)² = 1

**Loss value when using log loss:**

Before plugging in the values for loss equation, we can have a look at how the graph of  **_log(x)_** looks like.

![](https://miro.medium.com/max/575/0*gFI_OkR7RfjiJkWe.png)

**Figure 3: f(x) = log(x)**

As seen from the above graph as  **x tends to 0,**  **_log(x) tends to -infinity._**

Therefore, loss value would be:

**_-(1 * log(0) + 0 * log(1) ) = tends to infinity !!_**

**_As seen above, loss value using MSE was much much less compared to the loss value computed using the log loss function._** **_Hence it is very clear to us that MSE doesn’t strongly penalize misclassifications even for the perfect mismatch!_**

However, if there is a perfect match between predicted values and actual labels both the loss values would be “0” as shown below.

Actual label:  **“1”**

Predicted:  **“1”**

**_MSE: (1 - 1)² = 0_**

**_Log loss: -(1 * log(1) + 0 * log(0)) = 0_**

**_Here we have shown that MSE is not a good choice for binary classification problems. But the same can be extended for multi-class classification problems given that target values are one-hot encoded._**

# MSE and problem of Non-Convexity in Logistic Regression.

In classification scenarios, we often use gradient-based techniques(Newton Raphson, gradient descent, etc ..) to find the optimal values for coefficients by minimizing the loss function. Hence if the loss function is not convex, it is not guaranteed that we will always reach the global minima, rather we might get stuck at local minima.

![](https://miro.medium.com/max/579/0*vTqhydzUxLusIx9v.png)

Figure 4: Convex and non-Convex functions

Before diving deep into why MSE is not a convex function when used in logistic regression, first, we will see what are the conditions for a function to be convex.

A  [real-valued function](https://en.wikipedia.org/wiki/Real-valued_function)  defined on an  [_n_-dimensional interval](https://en.wikipedia.org/wiki/Interval_(mathematics)#Multi-dimensional_intervals)  is called  **convex**  if the  [line segment](https://en.wikipedia.org/wiki/Line_segment)  between any two points on the  [graph of the function](https://en.wikipedia.org/wiki/Graph_of_a_function)  lies above or on the graph.

![](https://miro.medium.com/max/875/0*Sifm_z3Nx-YEBvws.png)

Figure 5: Convexity of a function

If  **f**  is twice differentiable and the domain is the real line, then we can characterize it as follows:

**f** is  convex if and only if  **f ”(x) ≥ 0** for all  **_x_**_._ **_Hence if we can show that the double derivative of our loss function is ≥ 0 then we can claim it to be convex._** _For more details, you can refer to_ [_this video._](https://www.youtube.com/watch?v=u8JrE9JlZPM)

Now we mathematically show that the MSE loss function for logistic regression is non-convex.

For simplicity, let's assume we have one feature  **“x”**  and  **“binary labels”**  for a given dataset. In the below image  **f(x) = MSE**  and ŷ is the predicted value obtained after applying sigmoid function.

![](https://miro.medium.com/max/875/1*IOVd6NRlWZvNTAZbk0oV0g.png)

Figure 6: MSE double derivative

From the above equation,  **ŷ * (1 - ŷ)**  lies between [0, 1]. Hence we have to check that if  **H(ŷ)**  is positive for all values of  **“x”**  or not, to be a convex function.

We know that  **y** can take two values **0** or **1.** Let’s check the convexity condition for both the cases.

![](https://miro.medium.com/max/875/1*IoY50BDq57x0SF16zRim-g.png)

Figure 7: Double derivate of MSE when y=0

So in the above case when y = 0, it is clear from the equation that when ŷ lies in the range  **[0, 2/3]**  the function  **H(ŷ)**  **≥ 0 and** when ŷ lies between  **[2/3, 1]**  the function  **H(ŷ) ≤ 0.** This shows the function is not convex.

![](https://miro.medium.com/max/875/1*oro7dr2nUZtUdZfopIZ_4A.png)

Figure 8: Double derivative of MSE when y=1

Now, when  **y = 1**, it is clear from the equation that when ŷ lies in the range  **[0, 1/3]**  the function  **H(ŷ)**  ≤  **0 and** when ŷ lies between  **[1/3, 1]**  the function  **H(ŷ) ≥ 0.** This also shows the function is not convex.

Hence, based on the convexity definition we have mathematically shown the MSE loss function for logistic regression is non-convex and not recommended.

Now comes the question of  **convexity of the “log-loss” function!!** We will mathematically show that log loss function is convex for logistic regression.

![](https://miro.medium.com/max/875/1*HpLPxK-4YJRDiqEHtBylkQ.png)

Figure 9: Double derivative of log loss

Theta: co-efficient of independent variable “x”.

As seen in the final expression(double derivative of log loss function) the squared terms are always ≥0 and also, in general, we know the range of  **e^x** is **(0, infinity).  _Hence the final term is always ≥0 implying that the log loss function is convex in such scenarios !!_**

[Source](https://towardsdatascience.com/why-not-mse-as-a-loss-function-for-logistic-regression-589816b5e03c)
