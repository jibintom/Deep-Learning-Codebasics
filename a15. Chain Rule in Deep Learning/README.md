# Backpropagation in Neural Networks

## Neural Networks from scratch including math and python code

![](https://miro.medium.com/max/875/0*y4aag7gH0QbE2obB)

Photo by  [JJ Ying](https://unsplash.com/@jjying?utm_source=medium&utm_medium=referral)  on  [Unsplash](https://unsplash.com/?utm_source=medium&utm_medium=referral)

# Introduction

Have you ever used a neural network an wondered how the math behind it works? In this blogpost, we will derive forward- and back-propagation from scratch, write a neural network python code from it and learn some concepts of linear algebra and multivariate calculus along the way.

I will start off by explaining some linear algebra fundamentals. If you are proficient enough in this, you can skip the next part.

# Vectors and matrices

Simple numbers (scalars) are written in small letters.

![](https://miro.medium.com/max/155/1*JTyDPoFJbZ8M0UegyE101A.png)

Image by author

Vectors are denoted by bold letters and they are column vectors by default.

Names of row vectors are also written in bold but with a capital T for transposed.

![](https://miro.medium.com/max/304/1*B3iLECuw8iCsxxGZaSA4jQ.png)

Image by autor

Matrices are denoted in bold capital letters.

![](https://miro.medium.com/max/369/1*xmhMAxcTowADMJ2ddaq1Ag.png)

Image by autor

When transposing a matrix, we switch rows and columns.

![](https://miro.medium.com/max/308/1*XCMjRya4rVGw0yygwfmnag.png)

Image by author

The dimension of a Matrix or vector is:

( number of rows, number of columns)

![](https://miro.medium.com/max/275/1*KDfZ7DUXTC4r9Htjp6m3iw.png)

![](https://miro.medium.com/max/255/1*aortxRjuwPOPqEgdVHe1sw.png)

![](https://miro.medium.com/max/278/1*aLfMZr4YGg1b18Rt5P2bsw.png)

Image by author

# Dot product ⋅

When multiplying two vectors or matrices with the dot product, the number of columns of the left must match the number of rows of the right vector or matrix in the multiplcation.

![](https://miro.medium.com/max/875/1*gtELFbAf4CStRjEB-jQ9Pg.png)

(Image by author)

The dimension of the result can be calculated as follows:

![](https://miro.medium.com/max/356/1*0VcbMGF7sB6yXNIC38gQvA.png)

Image by author

The “inner” dimension in this multiplication must match and disappear in the output dimension.

# Elementwise product (Hadamard product) ⊙

When multiplying two vectors or matrices using the elementwise product, they have to have the same dimensions.

![](https://miro.medium.com/max/650/1*RXIyeTaXiM3CqkhE10RhCQ.png)

(Image by author)

# Matrix calculus

We also need calculus with vectors and matrices. We use the concept of a Jacobian. A Jacobian is simply a vector or matrix of derivatives.

The derivative of a scalar valued fuction with respect to a vector  **𝑥** will be defined as follows:

![](https://miro.medium.com/max/875/1*9rro94a42uM4wsFBSWOseg.png)

Image by author

We define the derivative of a function with an ouput vector  **𝑓** with respect to a single variable x as follows:

![](https://miro.medium.com/max/261/1*-HmcPfftuPBmGvTS7TId4g.png)

Image by author

The derivative of a function with an output vector and a vector of input variables is defined the following way:

![](https://miro.medium.com/max/875/1*0WAgZSbd95r2LvSWacdxng.png)

Image by author

# Vector chain rule

The vector chain rule looks very similar to the scalar chain rule. In this exmaple,vector  **𝑓**  is a function of vector  **𝑔**  and vector  **𝑔**  itself is a function of vector  **𝑥**. So the derivative of  **𝑓**(**𝑔**(**𝑥**)) with respect to  **𝑥**  is calculated the following way:

![](https://miro.medium.com/max/875/1*VFCvUG00Pq8i6ilO8g020w.png)

Image by author

# Neuron with single input vector

![](https://miro.medium.com/max/875/1*wqpDszybT3E-8FiQh7-Q8g.png)

Image by author

**𝑥**: Is the input sample into the neuron. In the above image, it has 3 features. This could be for example the height(1.75 m), weight(80kg) and age(30) of one human. Note that we only input one sample (one human in our example) into the neuron at once.

𝑎: Is called the activation and it is the output of the neuron. It is a prediction the neuron makes according to one input sample. We can choose ourselfs what our neuron should predict. A useful thing to predict in our human example could be the BMI.

**𝑤**: Weights of the neuron. It has seperate weights for each input feature of vector  **𝑥**.

𝑏:  bias of the neuron. One neuron has only one bias.

𝑦: Is the true or target value of the output of the network. We want 𝑎 to be as close to 𝑦 as possible. We therefore change the weights  **𝑤** and the bias 𝑏 in the training phase.

𝐿: Is called the Loss. The closer 𝑎 is to 𝑦, the smaller the loss is and therefore the better. Hence we want to minimize the loss. The most common way to minimize the loss in neural network is gradient descent. I will use the “mean squared error” loss over the course of this blogpost.

## Gradient descent

During training  **w** is updated in the following way.

![](https://miro.medium.com/max/336/1*SPa3nRLIV85jV0OIdTbkug.png)

Image by author

Where  **w´** is the new updated weight vector and 𝜆 is the learning rate. 𝜆 is a hyperparameter and can be chosen freely. Normally it is around 0.05.

![](https://miro.medium.com/max/281/1*a01NF4f7iYn891edzUU7DQ.png)

Image by author

∇𝐿(**w**) is the gradient of the loss with respect to the weights. It is simply the Transpose of the derivative of the loss 𝐿 with respect to the weights. In general the gradient of a function with respect to some varaibles is always the transpose of the according derivative.

![](https://miro.medium.com/max/298/1*m9QD2pXXQ5kmSnBb6_DXoQ.png)

Image by author

𝑏´ is the new updated bias. ∇𝐿(𝑏) is the gradient of the loss with respect to the weights.

![](https://miro.medium.com/max/266/1*VlQiwwG0wkig_hLQ7pU7Cw.png)

Image by author

In this case it does not matter, that the gradient is the transpose of the derivative, because it is a single number anyway and transposing a single number does not change it.

## Calculating gradients

Applying vector chain rule:

![](https://miro.medium.com/max/250/1*O4A5gFgOrQ6x_V0q1iwuBw.png)

![](https://miro.medium.com/max/239/1*mspJc1gWoPl-hwkY3xUEEQ.png)

![](https://miro.medium.com/max/464/1*YmFCmqG2frg-cvf_aXfVvw.png)

Image by author

Derivative with respect to the weights:

![](https://miro.medium.com/max/875/1*ogOq7xyOpXFz6pJtn_xEXw.png)

![](https://miro.medium.com/max/516/1*0oRH_s22RYV4iNPOlFa4LA.png)

Image by author

**𝑥** does not depend on  **w**, therefore it be treated as a constant and the derivative must only be taken with respect to  **w.**

Derivative with respect to the bias:

![](https://miro.medium.com/max/616/1*NeD23fqEuZSaoVTBgupo7w.png)

![](https://miro.medium.com/max/480/1*QyQc6znlwrYVV8TKAvKg7g.png)

Image by author

## **Gradient descent**

![](https://miro.medium.com/max/875/1*90vlidLznSflVUOqlEDapQ.png)

![](https://miro.medium.com/max/855/1*myt152f1Qi6A9GQ-Z1gHYw.png)

Image by author

## **Code**

Code by author

Code to test our neuron:

code by author

We then test our data with a dummy dataset to see, if it works. Each sample has 3 features. The desired output is 1, if the first feature is 1. The other 2 features do not matter for the outcome. So we expect the first weight to become 1, while the other weights and the bias will go to 0.

![](https://miro.medium.com/max/875/1*0wcJsvhXxiacJ1CWMgsOJw.png)

Plots showing the evolution of weights, bias and loss during training (Image by author)

We test the trained network on a sample it has never seen during training to see how well it generalizes to unseen data.

Test:   
predicted y: 0.9992918862001037, true y: 1

The ouput from our test sample shows, that we got very close to the true value.

# Ability to train on multiple samples at the same time

## Forward pass

each data point has size 3:

we have datapoints 𝑥1, 𝑥2, …

The dots denote, that the number of datapoints (𝑚) we can input in the neuron is variable.

![](https://miro.medium.com/max/873/1*5WVz2wQAKuFMkoac4qfImA.png)

Image by author

To get the input matrix, we stack the input vectors vertically:

![](https://miro.medium.com/max/350/1*xQLLkMaNbxKwokvSW_4-WQ.png)

Image by author

We have to adjust the loss as following:

![](https://miro.medium.com/max/339/1*ACGFRu7w-Mknsi8SIy_wsA.png)

Image by author

where 𝑚 is the number of datapoints we use.

**𝑎,** the activation vector is then calculated the following way:

![](https://miro.medium.com/max/686/1*OUUMC2dYkSzlXjY-JrdYRQ.png)

Image by author

![](https://miro.medium.com/max/875/1*FBC7_ktXJVbcSxpMrNmvKA.png)

Image by author

## Backward pass

Applying vector chain rule:

![](https://miro.medium.com/max/156/1*mfD4c0FIFaYqhYJuwQ71lA.png)

![](https://miro.medium.com/max/149/1*qW7wLyHLFKPgdatA22A1aQ.png)

Image by author

Calculating the derivatives.

![](https://miro.medium.com/max/875/1*CCRlWeMemcLaXrop8n0w_g.png)

Image by author

With respect to the weights:

![](https://miro.medium.com/max/674/1*zq4CS7U5fJWNHJzaM8zQXA.png)

![](https://miro.medium.com/max/364/1*xFh_Jw0DhnQ9cl3yR1FrOg.png)

Image by author

With respect to the bias:

![](https://miro.medium.com/max/711/1*BawCeqJGiZrmIMWKehnrDg.png)

![](https://miro.medium.com/max/583/1*dunkum_JqaTBqofQwt3J5A.png)

Image by author

## Gradient descent

![](https://miro.medium.com/max/659/1*oia1Eq-ACiipASUjSr-Bmw.png)

![](https://miro.medium.com/max/601/1*zO5oOKBhW2g-OCyHwCZJSg.png)

Image by author

## Code

Code by author

## Stochastic gradient descent

Stochastic gradient descent means, that we train on all training samples at the same time.

We can see the same evolution of weights, bias and loss as in the neuron with a single input.

![](https://miro.medium.com/max/875/1*DYKVLW36x1we6lohLs036g.png)

Image by author

We can see below, that the predicted output is very close to the true y:

Test:   
predicted y: 0.9999991060923578, true y: 1

## Minibatch gradient descent

Minibatch gradient descent means, that we train the neuron on a subset of the training set at the same time.

We have to write a function to generate random batches.

Code by author

![](https://miro.medium.com/max/875/1*1VZAoz9Q-W_mWtKJQH2nDA.png)

Image by author

predicted y: 0.9956078874633946, true y: 1

# Adding activation functions

Activation functions are nonlinear functions.

3 very important activation functions and their derivatives:

tanh:

![](https://miro.medium.com/max/206/1*fjcVxM1iX4hdvZ31X6ILkg.png)

![](https://miro.medium.com/max/268/1*0BzgJhGLwDQ4kNMEgxjgaw.png)

Image by author

simoid:

![](https://miro.medium.com/max/161/1*pGm6WZTa5DeauaJPxm7P6Q.png)

![](https://miro.medium.com/max/271/1*cpvWv7pb7aEuWBmZXGxk2A.png)

Image by author

rectified linear unit:

![](https://miro.medium.com/max/250/1*9p8y9Vv7jTdHKsFJqaEr9g.png)

Image by author

## Forward pass

![](https://miro.medium.com/max/429/1*Ec4vzQBN4BpSaDnpkOYC2Q.png)

![](https://miro.medium.com/max/126/1*epbaxmTAK-M79MnJLg0SMg.png)

![](https://miro.medium.com/max/175/1*VgK4i0ARGA7cCUWEOV6lKw.png)

![](https://miro.medium.com/max/90/1*mUeP0VmGMqAisgXB74kuXA.png)

![](https://miro.medium.com/max/211/1*S4NwYNKJCZTJJrgN8lJP-Q.png)

![](https://miro.medium.com/max/875/1*kHZFnt_vBE9JJ89onaiBPQ.png)

Images by author

## Backwards pass

Applying vector chain rule:

![](https://miro.medium.com/max/206/1*zxq9BTlgqThFuDwiJl0HJg.png)

![](https://miro.medium.com/max/199/1*AIfHudoRRhejoCXnptyr8w.png)

Image by author

Calculating the derivatives:

![](https://miro.medium.com/max/180/1*B4443LKszZtDYFBkbEZJ0A.png)

![](https://miro.medium.com/max/875/1*9UuR3srcLIt2I4a2vBH7CQ.png)

Image by author

The np.diag function constructs a matrix from a vector where the diagonal of the matrix is equal to the vector and it is 0 off-diagonal.

With respect to the weights:

![](https://miro.medium.com/max/674/1*VhrMbR8wZmo3Km0myy5mqg.png)

![](https://miro.medium.com/max/564/1*qYWuQd3keAbQUrAWTNgJuQ.png)

Image by author

With respect to the bias:

![](https://miro.medium.com/max/724/1*RvkAVtgfuei91TxJ5k6-wg.png)

![](https://miro.medium.com/max/875/1*HG_MeTYVt-z0DrYhi_GGGA.png)

Image by author

## Code

Activation functions:

Code by author

Neuron:

Code by author

**Minibatch gradient descent**

![](https://miro.medium.com/max/875/1*ddWIAGeNJkr4-fr9ZzYSFA.png)

Image by author

The values, the weights and the bias converge to are different to the previous cases.

prediceted y: 0.9394357707018176, true y: 1

# **Deep Neural Networks**

**Forward pass**

![](https://miro.medium.com/max/875/1*zVvdGCNcBFMAjMCp4BdKjQ.png)

Image by author

The superscripts in parentheses denote the layer.

**_Layer 1 (= input layer):_**

![](https://miro.medium.com/max/228/1*kcSGqX6GicIpjF29LnpeUQ.png)

![](https://miro.medium.com/max/179/1*78SOjh3hA8VJbGUttmVkUA.png)

Image by author

**_Layer 2 (= output layer):_**

![](https://miro.medium.com/max/233/1*PLGdtE3LAPbiUZNzxQ9Czg.png)

![](https://miro.medium.com/max/138/1*RZ9gnJIQAbA5BNxb318l4A.png)

Image by author

**_Loss:_**

![](https://miro.medium.com/max/228/1*lnDnSf636GeX7FAdGXvilA.png)

Image by author

## **Backward pass**

**_Layer 1 (= input layer):_**

![](https://miro.medium.com/max/701/1*zPtP-tW6cVNTef8LgGS4hw.png)

![](https://miro.medium.com/max/686/1*xM9uTtU7fusm78X8ffLU4w.png)

Image by author

**_Layer 2 (= output layer):_**

![](https://miro.medium.com/max/536/1*cloqACTf02VgFsnvfE1PiA.png)

![](https://miro.medium.com/max/513/1*Zb2wvzQkDBhmX07eqj8Nhw.png)

Image by author

Calculating the derivatives of layer 2 poses a problem since we would have to take the partial derivatives of vectors with respect to matrices and matrices with respect to matrices. This would result in tensors, which would make it unnecessary complicated.

We do not use the vector chain rule but calculate the final derivatives of the loss with respect to weights and biases directly. To do this, we:

1.  Rewrite the forward pass in index notation. In this notation we can see what each element of the final derivative of the loss with respect to the weights and biases looks like.
2.  Calculate the derivatives.
3.  Work back to a vectorized form in order to write efficient code.

**_Layer 1 (= input layer):_**

![](https://miro.medium.com/max/264/1*prMTrGrfuwzz9zKLcWC2sg.png)

![](https://miro.medium.com/max/166/1*3NN_mh-ZM4zAk-eq0ZwXmw.png)

Image by author

**_Layer 2 (= output layer):_**

![](https://miro.medium.com/max/266/1*7meUzjF4XYsOWt3Y0avJWA.png)

![](https://miro.medium.com/max/166/1*0DvIhaYTs6F0466GqFuhHA.png)

Image by author

**_Loss:_**

![](https://miro.medium.com/max/228/1*lnDnSf636GeX7FAdGXvilA.png)

Image by author

## **Backward pass**

**_Layer 2 (= output layer):_**

To derive the formulas in index notation we can look at the computation graph. For example to derive the formula for the weights of layer 2, we construct a computation graph, where all variables that depend on each other are connected.

![](https://miro.medium.com/max/456/1*UKNitrTCxJ53r8pU519dmQ.png)

Image by author

We rewrite the same coputation graph in index notation.

![](https://miro.medium.com/max/455/1*JxvxKzajYzZeJX0vXyVKQA.png)

Image by author

We now want to want to know derivative of the loss with respect to each weight. For each weight we have to sum over all branches of the graph, where this weight occurs. As an example for weight 1 of layer 2:

![](https://miro.medium.com/max/689/1*iK_vySet34iKZeMOH5uldA.png)

Image by author

The derivatives of the loss with respect to all weights is then:

![](https://miro.medium.com/max/689/1*tHKXPvOVvVD59-OMEsEUqw.png)

Image by author

Written back in vectorized form:

![](https://miro.medium.com/max/394/1*zbB7o2B9n70YlKVhz4weBg.png)

Image by author

Note that by looking at the left hand side of the equation we can see the dimension our vectorized form must have in the end. This helps us construct the dot product the right way around and gives us informations where we must transpose.

![](https://miro.medium.com/max/625/1*E0HtYCEPV9O7J27wqgP_Yg.png)

Image by author

Written back in vectorized form:

![](https://miro.medium.com/max/380/1*kn5BPSd_M8rZpYK8qJ1hKw.png)

Image by author

**_Layer 1 (= input layer):_**

![](https://miro.medium.com/max/875/1*pJGiAq29TGZdb00_QDBwBw.png)

Image by author

Written back in vectorized form:

![](https://miro.medium.com/max/608/1*jGDxrkbKpo00MQZ6F0fDyQ.png)

Image by author

Regrading the bias:

![](https://miro.medium.com/max/875/1*O496R5DUzvhPfMeaRbY7SA.png)

Image by author

Written back in vectorized form:

![](https://miro.medium.com/max/659/1*dHbGoASGF_qZ44NaT6HC-g.png)

Image by author

To make the summation more clear:

![](https://miro.medium.com/max/799/1*GC0ft1Jg69AsO5pZf9jeRA.png)

Image by author

“axis = 1” means we are summing over the columns.

# General formula

To generalize the formulas we derived above we introduce the concept of input Loss into the neurons into a layer. The input error  **𝛿**(𝐿) into the last layer 𝐿 is:

![](https://miro.medium.com/max/529/1*HMjHUO6MbdrfUSrx7g18gA.png)

Image by author

In vectorized form:

![](https://miro.medium.com/max/334/1*Sp9_oUfUMuxkQchocYWDVw.png)

Image by author

For all other layers l of our network:

![](https://miro.medium.com/max/875/1*jfjWb1JwQUq-F6753SYdLw.png)

Image by author

In vectorized form:

![](https://miro.medium.com/max/555/1*IeJ8Xwrq53OJ0cBpqNyWsw.png)

Image by author

𝑓 denotes the nonlinear activation function. This could be for example sigmoid or relu. And 𝑓′ denotes the derivative of 𝑓.

**_derivative with respect to weight:_**

![](https://miro.medium.com/max/624/1*jr2FH8DXe7jKacQg7jUeoQ.png)

Image by author

In vectorized form:

![](https://miro.medium.com/max/728/1*Ep6h-2yu1Ay2nf_-NAhLFw.png)

Image by author

**_derivative with respect to bias:_**

![](https://miro.medium.com/max/569/1*zET4Ib1skMv1Ukj_8AQXmA.png)

Image by author

In vectorized form:

![](https://miro.medium.com/max/763/1*wUFrLrGIEnFpwrxgDYA0tw.png)

![](https://miro.medium.com/max/875/1*TSTgd2zg8wrJ_l_-1vo8hg.png)

Image by author

# **Final Code**

This Code can be used to create and train arbitrary deep neural networks. The list “layers” passed to the __init__ method can be changed and this changes the network. The lenght of the list is equal to the number of layers. The numbers are equal to the number of neurons in each layer. The first must be equal to the number of features.

Code by author

Testing our code:

Code by author

We once again look at the evoultion of the weights and the bias of the output layer and the evolution of the loss during training.

![](https://miro.medium.com/max/875/1*AC4vklAaHoYCJA8ACts6Ug.png)

Image by author

predited y: 0.9514167066315814, true y: 1

# **Conclusion**

If you are still with me, I congratulate you, this was a though ride. And don’t be desperate if you did not get it the first try. It took me more than a month to figure this all out.

Today you have learned how to derive the gradients for weights in arbitrary deep neural networks. You should now also be able to apply this knowledge to other proplems like linear regression, Recurrent neural networks and reinforcement learning to name a few examples.

But be aware of the fact that there exist very good autograd framework nowadays, which are capable of numerically calculating the gradients for you. I am nevertheless of the mind that deriving these formulas by hand once helps our general understanding of how neural networks operate and some problems that arise like vanishing and exploding gradients in vanilla RNN.

[Source](https://towardsdatascience.com/backpropagation-in-neural-networks-6561e1268da8)
