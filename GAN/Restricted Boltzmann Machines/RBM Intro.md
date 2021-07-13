# 1. Restricted Boltzmann Machines

## 1.1 Architecture

In my opinion RBMs have one of the easiest architectures of all the NN. ```RBMs``` consist of  one input/visible layer (v1,...,v6), one hidden layer (h1, h2) and corresponding biases vectors ```Bias a``` and ```Bias b```. The absence of an output layer is apparent.

An ```output``` layer is not necessary since the predictions are made differently as in regular ```feedforward``` networks.

<img src="https://miro.medium.com/max/875/1*HkaD1isPyQ2ijl9lxXkV3A.png"/>

## 1.2 Energy-based Model

One purpose of deep learning models is to encode dependencies between variables. The capturing of dependencies happen through associating of a ```scaler energy``` to each configuration of the variables, which serves as a measure of compatibility.

A higher enery means bad compatability. An energy based model tries always to minimize a predefined energy function. 

The energy function for the ```RBMs``` is defined as:

<img src="https://miro.medium.com/max/646/1*dJggFW3OL9Y420gWeJXMBw.png"/>

The value of enegy depends on the configurations of ```visible/input units```, ```hidden units``` and ```biases```. The training of ```RBM``` consists of parameters for given input values so that the energy reaches minimum.

## 1.3 Probabilitic Model

Restricted Boltzmann Machines are probabilistic. As opposed to assigning discrete values to the model assign probabilities to the model. At each point in time the ```RBM``` is in a certain state. The state refers to the values of neurons in the ```visible``` and ```hidden``` layers ```v``` and ```h```.

The ```probability``` that a certain state of ```v``` and ```h``` can be observed is given by:

<img src="https://miro.medium.com/max/333/1*Iq2Tn7aLAegiIg4sfkSgGA.png"/>

Here ```Z``` is called the ```‘partition function’``` that is the summation over all possible pairs of visible and hidden vectors.

In Physics, the joint distribution is known as ```Boltzmann Distribution``` which gives the probability that a particle can be observed in a certain state with energy ```E```. As in Physics we assign a probability to observe a state of ```v``` and ```h```, that depends on the overall energy of the model. 

Unfortunately it is very difficult to calculate the ```joint probability``` due to the huge number of possible combination of ```v``` and ```h``` in the partition function ```Z```.
Much easier is the calculation of the conditional probabilities of state ```h``` given state ```v``` and conditional probabilities of state ```v``` given the state ```h```: 

<img src="https://miro.medium.com/max/355/1*NxzVmlmv6KDqO2k77WnfnA.png"/>

It should be noticed beforehand that each neuron in a ```RBM``` can only exist in a binary state `0` or `1`. The most intresting factor is the probability that a ```hidden``` or ```visible``` neuron is in state ```1``` -- hence activated. Given an input vector ```v``` the probability for a single hidden neuron <i>```j```</i> being activated is:

<img src="https://miro.medium.com/max/725/1*yx_C_ItC8aCUYbHhCJQY5g.png"/>

Here is ```σ``` the Sigmoid function.  This equation is derived by applying the ```Bayes Rule``` to Eq.3 and a lot of expanding which will be not covered here.

Analogous the probability that a binary state of a visible neuron <i>```i```</i> is set to ```1``` is:

<img src="https://miro.medium.com/max/725/1*6BMmNqK8H3a_BFSq5K3j-A.png"/>

# 2. Collaborative Filtering with Restricyed Boltzmann Machines

## 2.1 Recognizing Latent Factors in the Data

Lets assume some people were asked to rate a set of movies on a scale of ```1–5 stars```. In ```classical factor analysis``` each movie could be explained in terms of a set of latent factors. For example, movies like <i>```Harry Potter and Fast and the Furious```</i> might have strong associations with a latent factors of fantasy and action. On the other hand users who like <i>```Toy Story and Wall-E```</i> might have strong associations with latent <i>```Pixar```</i> factor.

```RBMs``` are used to ```analyse``` and find out these underlying factors. After some epochs of the training phase the neural network has seen all ratings in the training date set of each user ```multiply times```. 

At this time the model should have learned the underlying ```hidden factors``` based on users preferences and corresponding collaborative movie tastes of all users.

The analysis of hidden factors is performed in a binary way. Instead of giving the model user ratings that are continues (e.g. 1–5 stars), the user simply tell if they liked (rating 1) a specific movie or not (rating 0). The binary rating values represent the inputs for the input/visible layer. Given the inputs the RMB then tries to discover latent factors in the data that can explain the movie choices. Each hidden neuron represents one of the latent factors. Given a large dataset consisting out of thousands of movies it is quite certain that a user watched and rated only a small amount of those. It is necessary to give yet unrated movies also a value, e.g. -1.0 so that the network can identify the unrated movies during training time and ignore the weights associated with them.

Lets consider the following example where a user likes Lord of the Rings and Harry Potter but does not like The Matrix, Fight Club and Titanic. The Hobbit has not been seen yet so it gets a -1 rating. Given these inputs the Boltzmann Machine may identify three hidden factors Drama, Fantasy and Science Fiction which correspond to the movie genres.

<img src="https://miro.medium.com/max/875/1*ZY4c980_7MfEMYTIi6jvTw.png"/>

Given the movies the ```RMB``` assigns a probability ```p(h|v) (Eq. 4)``` for each hidden neuron. The final binary values of the neurons are obtained by sampling from ```Bernoulli distribution``` using the probability `p`.

In this example only the hidden neuron that represents the genre Fantasy becomes activate. Given the movie ratings the Restricted Boltzmann Machine recognized correctly that the user likes Fantasy the most.

## 2.2 Using Latent Factors for Prediction

After the training phase the goal is to predict a ```binary rating``` for the movies that had not been seen yet. Given the training data of a specific user the network is able to identify the latent factors based on this users preference.

Since the latent factors are represented by the ```hidden neurons``` we can use ```p(v|h) (Eq. 5)``` and sample from ```Bernoulli distribution``` to find out which of the visible neurons now become active.

<img src="https://miro.medium.com/max/875/1*De0RDPU_XRqT0BMAVE4vqA.png"/>

Fig. 4 shows the new ratings after using the hidden neuron values for the inference. The network did identified Fantasy as the preferred movie genre and rated The Hobbit as a movie the user would like.

# 3. Training

The training of the ```RBM``` differs from the training of a regular neural network via ```SGD``` optimizer. 

## 3.1 Gibbs Sampling

The first part of the training is called Gibbs Sampling. Given an input vector v we are using p(h|v) (Eq.4) for prediction of the hidden values h.

Knowing the hidden values we use ```p(v|h) (Eq.5)``` for prediction of new input values ```v```. This process is repeated `k` times. After `k` iterations we obtain an other input vector ```v_k``` which was recreated from original input values `v_0`.

<img src="https://miro.medium.com/max/875/1*UMbNSJVSmAgqkVnQKA62yg.png"/>


## 3.2 Contrastive Divergence

The update of the weight matrix happens during the ```Contrastive Divergence``` step.
Vectors <b>```v_0```</b> and <b>```v_k```</b> are used to calculate the activation probabilities for hidden values <b>```h_0```</b> and <b>```h_k```</b>. 

The difference between the ```outer products``` of those probabilities with input vectors <b>```v_0```</b> and <b>```v_k```</b> results in the update matrix:

<img src="https://miro.medium.com/max/590/1*GUmQmq2nfgKwMcHDoYsxHA.png"/>

Using the update matrix we can update the weight matrix using <b>```Gradient ascent```</b>:

<img src="https://miro.medium.com/max/331/1*npmDgs0itN1wbUwn5zTB2g.png"/>