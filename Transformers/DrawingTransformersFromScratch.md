# Google's Transformer Paper from Scratch (Encoder Part)

Drawing Transformer from Scratch from the Google Paper ```"Attention Is All You Need"```.

<img src="https://miro.medium.com/max/1400/1*YBd5d5ysZ2myU7Nxxh4yTg.png"/>

```Transformer``` has an ```encoder``` and ```decoder``` part.

Here we have the Encoder part of the Transformer from the bottom-top fashion. This below animation shows the working of the Encoder part.

<img src="https://miro.medium.com/max/875/1*UH5SSuMy9y-BcBtvAUhTmQ.gif"/>

## Input

A Transformer takes as input a sequence of words, which are presented to the NN as  ```vectors```. In NLP tasks usually a voabulary or dictionary is used, in which each word is assigned a unique integer index. 

The index can be represented as so called ```one-hot``` vector, which is predominantly made up of zeros, with a single "one" value at the correct location. A simple ```one-hot``` word encoding for a small vocabulary of size ```ten``` is shown below: 

<img src="https://miro.medium.com/max/470/1*y5d-7nqdE-QIZJCngreBog.png">

Please note that the one-hot encoded vectors have the <b>same size</b> as the number of words in the vocabulary, which in real-world application is at least ```10.000```. Furthermore, all one-hot encodings have the same Euclidean distance of ```√2``` to each other.

## Word Embeddings

Next, we reduce the dimensionality of the ```one-hot encoded vectors``` by multiplying them with a so called ```"embedding matrix"```. Resulting vectors are called ```word embeddings```. The size of the word embeddings in the paper is ```512```.

The huge benefit of word embeddings is that words with similar meanings are put close to each other, e.g. the word “cat” and “kitty” end up having similar embedding vectors.

Please note that the ```“embedding matrix”``` is a normal matrix, just with a fancy name.

## Positional Encodings

All the words are presented to the Transformer simultanously. This is a huge difference to ```RNNs``` like ```LSTMs``` or ```GRUs``` which are presented to the NN one word at a time.

However, this means that the order in which words occur in the input sequence is lost. To address this, the Transformer adds a ```vector``` to each input embedding, thus injecting some information about the relative or absolute position.

## Keys and Queries

Finally, we multiply the word embeddings by matrices ```WQ``` and ```WK``` to obtain the ```query vectors``` and ```key vectors```, each of size 64.

All the components, are like this:

<img src="https://miro.medium.com/max/875/1*GDC-85VrLe-J8p-G3kEZCA.gif"/>

## Parallelization

All the ```word embeddings``` can be computed in parallel. Once we’ve got the embeddings, we also can simultaneously compute the query vectors and key vectors for all the embeddings. This pattern will continue throughout the whole architecture.

## Dot Products

We calculate the dot products for all possible combinations of ```query vectors``` and ```key vectors```. The result of a dot product is a single number, which in a later step will be used as a  ```weight factor```.

The weights factors tell us, how much two words at two different positions in the input sequence depend on each other. This is called <b>```Self Attention```</b>. The mechanism of ```self-attention``` allows the Transformer to learn difficult dependencies even between ```distant positions```.

<img src="https://miro.medium.com/max/875/1*GzF4v8w0rOFAczA-sI6RCQ.gif">

## Scaling

All the weight factors are divided by 8 (sqrt of the dimensions of the ```key vectors``` 64). The assumption is taken that during training the dot products can grow large in magnitude, thus pushing the ```softmax``` function into regions with extremely ```small gradients```. Dividing by 8 leads to having more <b>stable gradients</b>.

## Softmax

The scaled factors are put through a ```softmax function```, which normalizes them so they are all ```positive and sum up to 1.```

In the animation, scaling is performed for the weight factors belonging to the first word in the sentence, which is ```"The"```. ```Weight Factors``` belonging to the first word 
are the dot products
```python
q1 * k1, q1 * k2, q1 * k3, q1 * k4
```  

<img src="https://miro.medium.com/max/875/1*ctrN__xt86dvW7NX06yCaQ.gif"/>

Analogously, for the other words “car”, “is” and “blue” in our input sequence we get:

<img src="https://miro.medium.com/max/875/1*ttQAXZnlrcq4QPfWIkVSlA.gif"/>

This completes the calculation of the ```weight factors```.

## Values

Identical to the computation of the ```key vector``` and ```query vectors``` we obtain the ```value vectors``` by multiplying the word embeddings by matrix ```WV```. Again the size of the ```value vectors``` is 64.

## Weighting

Now, we multiply each ```value vector``` by its corresponding ```weight factor```. As mentioned before, this way we only keep the words we want to focus on, while irrelevant words are suppressed by weighting them by tiny numbers like  ```0.001```.

## Summation

Now we sum up all the weighted ```value vectors``` belonging to a word. This produces the output of the ```self-attention layer``` at this position.

In the next animation we depict the computation of the ```value vectors``` and their subsequent weighting and summation performed for the first word in the input sequence:

<img src="https://miro.medium.com/max/875/1*x3NcwCBExL-_5UVPQraizg.gif"/>
Analogously for the other words “car”, “is”, “blue” in our input sequence, we get:

<img src="https://miro.medium.com/max/875/1*I3yHUigmcykbl8EEdpJopw.gif"/>

That concludes the ```self-attention calculation```. The output of the ```self-attention``` layer can be considered as a <b>context enriched word embedding</b>. Depending on the context, a word can have different meanings:
- I like fresh, crisp <b>```fall```</b> weather.
- Don't <b>```fall```</b> on your way to the tram.

Please note that the ```embeddings matrix``` at the bottom is operating on single words only. Hence for both sentences, we would wrongly obtain the ```same``` embedding vector. The ```self-attention layer``` is taking this into consideration.

### Short Sentences

The length of the input sequences is supposed to be fixed in length, basically it is the length of the longest sentence in the training data.  Hence, a parameter defines the maximum length of a sequence that the Transformer can accept.

```Sequences``` that are greater than this length are truncated. Shorter squences are padded with zeros. However, padded words are not supposed to contribute to the ```self-attention``` calculation. 

This is avoided by masking the corresponding words ```(setting them to -Inf)``` before the  ```softmax``` step in the ```self-attention``` calculation.
This infact sets their ```weight factors``` to ```0``` and ```suppresses``` them from the calculation.

<img src="https://miro.medium.com/max/875/1*uAn_WIagYnYYL-q3rfqhHQ.gif"/>

## Multi-head Self-Attention

Instead of performing a ```single self-attention calculation```, the Transformer performs ```multi-head self-attention```, each with different ```weight matrices```. 

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. The ```Transformer``` in the paper uses ```eight parallel attention heads```.

The outputs of the attention heads are ```concatenated``` and once again multiplied by an additional weight matrix ```WO```.

<img src="https://miro.medium.com/max/875/1*9SGL9LbtH1CWTkkHgV-Xpw.gif"/>

## Add & Normalize

The ```multi-head self-attention``` mechanism is the first submodule of the ```encoder```. It has a ```residual connection``` around it, and is followed by a ```layer normalization``` step. 

Layer Normalization just subtracts the mean of each vector and divides by its standard deviation.
<img src="https://miro.medium.com/max/875/1*FMRS9vf5B4aBf2c_e_F8gg.gif"/>

## Feed-Forward

The outputs of the ```self-attention``` layer are fed to a fully connected ```feed-forward layer```. 

This consists of two linear transformations with a ```ReLU``` activation in between. The dimensionality of input and output is ```512```, and the inner-layer has dimensionality ```2048```.

The exact same feed-forward network is indpendently applied to each position which is for each word in the ```inner sequence```.

Now, employ a ```residual connection``` around the ```Feed Forward``` layer, followed by ```Layer Normalization```.

<img src="https://miro.medium.com/max/875/1*2KZQBhtv2l2Z5or5IFUELg.gif"/>

## Stack of Encoders 

The entire encoding component is a ```stack``` of ```six encoders```. The encoders are all identical in structure, yet they do not share ```weights```.
<img src="https://miro.medium.com/max/875/1*hTuCAGdXQT-DV_--RW3iYg.gif"/>