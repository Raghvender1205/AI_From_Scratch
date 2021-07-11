# AutoEncoders

AutoEncoders are an unsupervised learning in which we leverage neural networks for the task of ```representation learning```.

Design a Neural Network architecture such that we impose ```bottleneck``` in the network which forces a ```compressed``` knowledge representation of the original input. 

If the input features were each independent of one another, this ```compression``` and ```subsequent reconstruction``` would be a very difficult task. However, if some sort of structure exists in the data (ie. correlations between input features), this structure can be learned and consequently leveraged when forcing the input through the network's ```bottleneck```.

<img src="https://www.jeremyjordan.me/content/images/2018/03/Screen-Shot-2018-03-06-at-3.17.13-PM.png"/>

We can take an unlabelled dataset and frame it as a supervised learning problem tasked with outputting ```x^```,  a ```reconstruction of the original  input x```. This network can be trained by minimizing the reconstruction error 
<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow class="MJX-TeXAtom-ORD">
    <mi class="MJX-tex-caligraphic" mathvariant="script">L</mi>
  </mrow>
  <mrow>
    <mo>(</mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>x</mi>
      <mo>,</mo>
      <mrow class="MJX-TeXAtom-ORD">
        <mover>
          <mi>x</mi>
          <mo stretchy="false">&#x005E;<!-- ^ --></mo>
        </mover>
      </mrow>
    </mrow>
    <mo>)</mo>
  </mrow>
</math>, which measures the difference between the original input and the consequent reconstruction.

The ```Bottleneck``` is a key attribute of our network design, without the presence of an information bottleneck, our network could easily learn to simply memorize the input values by passing these values along the ```Network```.

<img src="https://www.jeremyjordan.me/content/images/2018/03/Screen-Shot-2018-03-06-at-6.09.05-PM.png"/>

A ```bottleneck``` constrains the amount of information that can traverse the full network, forcing a learned ```compression``` of the input data.

The ideal ```AutoEncoder``` model balances the following:
- Sensitive to the inputs enough to accurately build a ```reconstruction```.
- Insensitive enough to the inputs that the model doesn't simply memorize or overfit the training data.

This trade-off forces the model to maintain only the variations in the data required to reconstruct the input without holding on to redundancies within the input.

For most cases, this involves constructing a loss function where one term encourages our model to be sensitive to the inputs (ie. reconstruction loss <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow class="MJX-TeXAtom-ORD">
    <mi class="MJX-tex-caligraphic" mathvariant="script">L</mi>
  </mrow>
  <mrow>
    <mo>(</mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>x</mi>
      <mo>,</mo>
      <mrow class="MJX-TeXAtom-ORD">
        <mover>
          <mi>x</mi>
          <mo stretchy="false">&#x005E;<!-- ^ --></mo>
        </mover>
      </mrow>
    </mrow>
    <mo>)</mo>
  </mrow>
</math> and a second term discourages memorization/overfitting (ie. an added regularizer).

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow class="MJX-TeXAtom-ORD">
    <mi class="MJX-tex-caligraphic" mathvariant="script">L</mi>
  </mrow>
  <mrow>
    <mo>(</mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>x</mi>
      <mo>,</mo>
      <mrow class="MJX-TeXAtom-ORD">
        <mover>
          <mi>x</mi>
          <mo stretchy="false">&#x005E;<!-- ^ --></mo>
        </mover>
      </mrow>
    </mrow>
    <mo>)</mo>
  </mrow>
  <mo>+</mo>
  <mi>r</mi>
  <mi>e</mi>
  <mi>g</mi>
  <mi>u</mi>
  <mi>l</mi>
  <mi>a</mi>
  <mi>r</mi>
  <mi>i</mi>
  <mi>z</mi>
  <mi>e</mi>
  <mi>r</mi>
</math>

We'll typically add a scaling parameter in front of the regularization term so that we can adjust the trade-off between the two objectives.

## UnderComplete AutoEncoder

The simplest architecture for constructing an ```AutoEncoder``` is to contrain the number of nodes present in the ```hidden layers``` of the network, limiting the amount the information that can flow through the network.

By penalizing the network according to the ```reconstruction error```, our model can ```learn``` the most important attributes of the input data and how to best reconstruct the original input from an ```"encoded"``` state. Ideally, this encoding will ```learn and describe latent attributes of the input data.```

<img src="https://www.jeremyjordan.me/content/images/2018/03/Screen-Shot-2018-03-07-at-8.24.37-AM.png"/>

Because neural networks are capable of learning ```nonlinear relationships```, this can be thought of as a ```non-linear``` generalization of ```PCA```.

Whereas ```PCA``` attempts to discover a ```lower dimensional hyperplane``` which describes the original data, ```autoencoders``` are capable of learning nonlinear manifolds .

<img src="https://www.jeremyjordan.me/content/images/2018/03/Screen-Shot-2018-03-07-at-8.52.21-AM.png"/>

For ```Higher dimensional data```, ```autoencoders``` are capable of learning a complex representation of the data which can be used to describe the observations in a lower dimensionality and corresponding decoded into the original input space.

<img src="https://www.jeremyjordan.me/content/images/2018/03/LinearNonLinear.png"/>

An ```undercomplete autoencoder``` has no explicit regularization term - we simply train our model according to the reconstruction loss. Thus, our only way to ensure that the model isn't memorizing the input data is the ensure that we've sufficiently restricted the number of nodes in the hidden layer(s).

For ```Deep AutoEncoders```, we must also be aware of the ```capacity``` of our ```Encoder``` and ```Decoder``` models. Even if the ```bottleneck layer``` is only one hidden node

It is still possible for the model to memorize the input data provided that the ```Encoder``` and ```Decoder``` models have sufficient capacity to learn some arbitrary function which can map the data to an index.

Given the fact that we'd like our model to discover ```latent attributes``` within our data, it's important to ensure that the ```autoencoder``` model is not simply learning an efficient way to memorize the training data.

Similar to ```supervised``` learning problems, we can employ forms of regularization to the network in order to encourage good generalization properties

## Sparse AutoEncoder

```Sparse AutoEncoder``` offers an alternative method for introducing an information bottleneck without ```repairing``` a reduction in the number of nodes in the ```hidden layers```. Rather, we'll construct our ```loss function``` such that we penalize ```activations``` within a layer. 

For an observation, we'll encourage our network to learn an ```encoding and decoding``` which only relies on <b>activating a small number of neurons</b>. It's worth noting that this is a different approach towards ```regularization```, as we normally regularize the weights of a network, not the activations.

It's important to note that the ```individual nodes``` of a trained model which activate are ```data-dependent```, different inputs will result in ```activations of different nodes``` through the network. 

<img src="https://www.jeremyjordan.me/content/images/2018/03/Screen-Shot-2018-03-07-at-1.50.55-PM.png"/>

One result of this fact is that we allow our network to ```sensitize individual hidden layer nodes toward specific attributes of the input data```. Whereas an ```undercomplete autoencoder``` will use the entire network for every observation, a ```sparse autoencoder``` will be forced to selectively ```activate``` regions of the network depending on the input data. 

As a result, we've limited the network's capacity to ```memorize``` the input data without limiting the networks capability to extract features from the data.

This allows us to consider the ```latent state representation``` and ```regularization``` of the network separately such that we can choose a ```latent state representation``` (ie. encoding dimensionality) in accordance with what makes sense given the context of the data while imposing ```regularization``` by the ```sparsity constraint```.

There are two main ways by which we can impose this ```sparsity constraint```; both involve measuring the ```hidden layer activations``` for each training batch and adding some term to the ```loss``` function in order to penalize ```excessive``` activations. These terms are:
- <b>L1 Regularization</b> :- We can add a term to our ```loss function``` that penalizes the absolute value of the ```vector of activations a``` in layer ```h``` for observation ```i```, scaled by a tuning parameter ```λ```.
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow class="MJX-TeXAtom-ORD">
    <mi class="MJX-tex-caligraphic" mathvariant="script">L</mi>
  </mrow>
  <mrow>
    <mo>(</mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>x</mi>
      <mo>,</mo>
      <mrow class="MJX-TeXAtom-ORD">
        <mover>
          <mi>x</mi>
          <mo stretchy="false">&#x005E;<!-- ^ --></mo>
        </mover>
      </mrow>
    </mrow>
    <mo>)</mo>
  </mrow>
  <mo>+</mo>
  <mi>&#x03BB;<!-- λ --></mi>
  <munder>
    <mo movablelimits="false">&#x2211;<!-- ∑ --></mo>
    <mi>i</mi>
  </munder>
  <mrow class="MJX-TeXAtom-ORD">
    <mrow>
      <mo>|</mo>
      <mrow class="MJX-TeXAtom-ORD">
        <msubsup>
          <mi>a</mi>
          <mi>i</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mrow>
              <mo>(</mo>
              <mi>h</mi>
              <mo>)</mo>
            </mrow>
          </mrow>
        </msubsup>
      </mrow>
      <mo>|</mo>
    </mrow>
  </mrow>
</math>

- <b>KL-Divergence</b>:- In essence, KL-divergence is a measure of the difference between two ```probability ditributions```. We can define a ```sparsity parameter ρ ```which denotes the average activation of a neuron over a collection of samples. This expectation can be calculated as 

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow class="MJX-TeXAtom-ORD">
    <msub>
      <mrow class="MJX-TeXAtom-ORD">
        <mrow class="MJX-TeXAtom-ORD">
          <mover>
            <mi>&#x03C1;<!-- ρ --></mi>
            <mo stretchy="false">&#x005E;<!-- ^ --></mo>
          </mover>
        </mrow>
      </mrow>
      <mi>j</mi>
    </msub>
  </mrow>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mi>m</mi>
  </mfrac>
  <munder>
    <mo movablelimits="false">&#x2211;<!-- ∑ --></mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>i</mi>
    </mrow>
  </munder>
  <mrow class="MJX-TeXAtom-ORD">
    <mrow>
      <mo>[</mo>
      <mrow class="MJX-TeXAtom-ORD">
        <msubsup>
          <mi>a</mi>
          <mi>i</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mrow>
              <mo>(</mo>
              <mi>h</mi>
              <mo>)</mo>
            </mrow>
          </mrow>
        </msubsup>
        <mrow>
          <mo>(</mo>
          <mi>x</mi>
          <mo>)</mo>
        </mrow>
      </mrow>
      <mo>]</mo>
    </mrow>
  </mrow>
</math>

where the subscript ```j``` denotes the specific neuron in layer ```h```, summing the activations for ```m``` training observations denoted individually as ```x```.

- In essence, by constraining the ```average activation of a neuron``` over a collection of samples we're encouraging neurons to only fire for a ```subset``` of the observations. We can describe ```ρ``` as a Bernoulli random variable distribution such that we can leverage the ```KL divergence``` (expanded below) to compare the ```ideal distribution ρ``` to the observed distributions over all ```hidden layer nodes ρ^```.

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow class="MJX-TeXAtom-ORD">
    <mi class="MJX-tex-caligraphic" mathvariant="script">L</mi>
  </mrow>
  <mrow>
    <mo>(</mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>x</mi>
      <mo>,</mo>
      <mrow class="MJX-TeXAtom-ORD">
        <mover>
          <mi>x</mi>
          <mo stretchy="false">&#x005E;<!-- ^ --></mo>
        </mover>
      </mrow>
    </mrow>
    <mo>)</mo>
  </mrow>
  <mo>+</mo>
  <munder>
    <mo movablelimits="false">&#x2211;<!-- ∑ --></mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>j</mi>
    </mrow>
  </munder>
  <mrow class="MJX-TeXAtom-ORD">
    <mi>K</mi>
    <mi>L</mi>
    <mrow>
      <mo>(</mo>
      <mrow class="MJX-TeXAtom-ORD">
        <mi>&#x03C1;<!-- ρ --></mi>
        <mrow class="MJX-TeXAtom-ORD">
          <mo stretchy="false">|</mo>
        </mrow>
        <mrow class="MJX-TeXAtom-ORD">
          <mo stretchy="false">|</mo>
        </mrow>
        <mrow class="MJX-TeXAtom-ORD">
          <msub>
            <mrow class="MJX-TeXAtom-ORD">
              <mrow class="MJX-TeXAtom-ORD">
                <mover>
                  <mi>&#x03C1;<!-- ρ --></mi>
                  <mo stretchy="false">&#x005E;<!-- ^ --></mo>
                </mover>
              </mrow>
            </mrow>
            <mi>j</mi>
          </msub>
        </mrow>
      </mrow>
      <mo>)</mo>
    </mrow>
  </mrow>
</math>

The ```KL divergence``` between two Bernoulli distributions can be written as

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <munderover>
    <mo movablelimits="false">&#x2211;<!-- ∑ --></mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>j</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mrow class="MJX-TeXAtom-ORD">
      <mrow class="MJX-TeXAtom-ORD">
        <msup>
          <mi>l</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mrow>
              <mo>(</mo>
              <mi>h</mi>
              <mo>)</mo>
            </mrow>
          </mrow>
        </msup>
      </mrow>
    </mrow>
  </munderover>
  <mrow class="MJX-TeXAtom-ORD">
    <mi>&#x03C1;<!-- ρ --></mi>
    <mi>log</mi>
    <mo>&#x2061;<!-- ⁡ --></mo>
    <mfrac>
      <mi>&#x03C1;<!-- ρ --></mi>
      <mrow class="MJX-TeXAtom-ORD">
        <mrow class="MJX-TeXAtom-ORD">
          <msub>
            <mrow class="MJX-TeXAtom-ORD">
              <mrow class="MJX-TeXAtom-ORD">
                <mover>
                  <mi>&#x03C1;<!-- ρ --></mi>
                  <mo stretchy="false">&#x005E;<!-- ^ --></mo>
                </mover>
              </mrow>
            </mrow>
            <mi>j</mi>
          </msub>
        </mrow>
      </mrow>
    </mfrac>
  </mrow>
  <mo>+</mo>
  <mrow>
    <mo>(</mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mn>1</mn>
      <mo>&#x2212;<!-- − --></mo>
      <mi>&#x03C1;<!-- ρ --></mi>
    </mrow>
    <mo>)</mo>
  </mrow>
  <mi>log</mi>
  <mo>&#x2061;<!-- ⁡ --></mo>
  <mfrac>
    <mrow class="MJX-TeXAtom-ORD">
      <mn>1</mn>
      <mo>&#x2212;<!-- − --></mo>
      <mi>&#x03C1;<!-- ρ --></mi>
    </mrow>
    <mrow class="MJX-TeXAtom-ORD">
      <mn>1</mn>
      <mo>&#x2212;<!-- − --></mo>
      <mrow class="MJX-TeXAtom-ORD">
        <msub>
          <mrow class="MJX-TeXAtom-ORD">
            <mrow class="MJX-TeXAtom-ORD">
              <mover>
                <mi>&#x03C1;<!-- ρ --></mi>
                <mo stretchy="false">&#x005E;<!-- ^ --></mo>
              </mover>
            </mrow>
          </mrow>
          <mi>j</mi>
        </msub>
      </mrow>
    </mrow>
  </mfrac>
</math>

This loss term is visualized below for an ```ideal distribution of ρ=0.2```, corresponding with the minimum ```(zero) penalty``` at this point.

<img src="https://www.jeremyjordan.me/content/images/2018/03/KLPenaltyExample-1.png"/>

## Denoising AutoEncoders

We would like our ```AutoEncoders``` to be sensitive enough to recreate the original observation but ```insensitive``` enough to the training data such that the model learns a ```generalizable``` encoding and decoding.

Another approach towards developing a ```generalizable model``` is to slightly ```corrupt``` the input data but still maintain the ```uncorrupted data``` as our target output.

<img src="https://www.jeremyjordan.me/content/images/2018/03/Screen-Shot-2018-03-09-at-10.20.44-AM.png"/>

With this approach, <b>our model isn't able to simply develop a mapping which memorizes the training data because our input and target output are no longer the same</b>. 

Rather, the model learns a ```vector field``` for mapping the input data towards a ```lower-dimensional manifold```, if this ```manifold``` accurately describes the natural data, we've effectively ```"canceled out"``` the added ```noise```.

<img src="https://www.jeremyjordan.me/content/images/2018/03/Screen-Shot-2018-03-09-at-10.12.59-PM.png"/>

The above figure visualizes the vector field described by comparing the ```reconstruction of x``` with the original value of ```x```. The yellow points represent training examples prior to the addition of ```noise```. As you can see, the model has learned to adjust the ```corrupted input``` towards the learned ```manifold```.

It's worth noting that this ```vector field``` is typically only well behaved in the regions where the model has observed during training. In areas far away from the natural data distribution, the ```reconstruction error``` is both <b>large and does not always point in the direction of the true distribution</b>.

<img src="https://www.jeremyjordan.me/content/images/2018/03/Screen-Shot-2018-03-10-at-10.17.44-AM.png"/>

## Categorical AutoEncoders

One would expect that for ```very similar inputs, the learned encoding would also be very similar```. We can explicitly train our model in order for this to be the case by requiring that the <b><i>derivative of the hidden layer activations are small</b></i> with respect to the input.

In other words, for ```small changes``` to the input, we should still maintain a very similar ```encoded state```. This is quite similar to a ```denoising autoencoder``` in the sense that these ```small perturbations``` to the input are essentially considered ```noise and that we would like our model to be robust against noise```.

```Denoising autoencoders``` make the ```reconstruction``` function (ie. decoder) resist small but ```ﬁnite-sized perturbations``` of the input, while ```contractive autoencoders``` make the ```feature extraction``` function (ie. encoder) resist ```infinitesimal perturbations``` of the input.

Because we're explicitly encouraging our model to learn an ```encoding``` in which similar inputs have ```similar encodings```, we're essentially ```forcing the model``` to learn how to <i><b>contract</i></b> a neighborhood of inputs into a ```smaller neighborhood of outputs```.

Notice how the slope (ie. derivative) of the reconstructed data is essentially ```zero``` for local neighborhoods of input data.

<img src="https://www.jeremyjordan.me/content/images/2018/03/Screen-Shot-2018-03-10-at-12.25.43-PM.png"/>

We can accomplish this by constructing a ```loss term``` which penalizes ```large derivatives of our hidden layer activations``` with respect to the input training examples, essentially penalizing instances where a small change in the input leads to a large change in the ```encoding space```.

In fancier mathematical terms, we can craft our ```regularization``` loss term as the ```squared Frobenius norm ∥A∥F of the Jacobian matrix J``` for the hidden layer activations with respect to the input observations.

A ```Frobenius norm is essentially an L2 norm for a matrix and the Jacobian matrix``` simply represents all ```first-order partial derivatives``` of a <b>vector-valued</b> function (in this case, we have a vector of training examples).

For ```m``` observations and ```n``` hidden layer nodes, we can calculate these values as follows:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow class="MJX-TeXAtom-ORD">
    <msub>
      <mrow>
        <mo symmetric="true">&#x2016;</mo>
        <mi>A</mi>
        <mo symmetric="true">&#x2016;</mo>
      </mrow>
      <mi>F</mi>
    </msub>
  </mrow>
  <mo>=</mo>
  <msqrt>
    <munderover>
      <mo movablelimits="false">&#x2211;<!-- ∑ --></mo>
      <mrow class="MJX-TeXAtom-ORD">
        <mi>i</mi>
        <mo>=</mo>
        <mn>1</mn>
      </mrow>
      <mi>m</mi>
    </munderover>
    <mrow class="MJX-TeXAtom-ORD">
      <munderover>
        <mo movablelimits="false">&#x2211;<!-- ∑ --></mo>
        <mrow class="MJX-TeXAtom-ORD">
          <mi>j</mi>
          <mo>=</mo>
          <mn>1</mn>
        </mrow>
        <mi>n</mi>
      </munderover>
      <mrow class="MJX-TeXAtom-ORD">
        <mrow class="MJX-TeXAtom-ORD">
          <msup>
            <mrow class="MJX-TeXAtom-ORD">
              <mrow>
                <mo>|</mo>
                <mrow class="MJX-TeXAtom-ORD">
                  <mrow class="MJX-TeXAtom-ORD">
                    <msub>
                      <mi>a</mi>
                      <mrow class="MJX-TeXAtom-ORD">
                        <mi>i</mi>
                        <mi>j</mi>
                      </mrow>
                    </msub>
                  </mrow>
                </mrow>
                <mo>|</mo>
              </mrow>
            </mrow>
            <mn>2</mn>
          </msup>
        </mrow>
      </mrow>
    </mrow>
  </msqrt>
</math>

<math display="block">
 <mstyle mathvariant="bold" mathsize="normal"><mi>J</mi></mstyle><mo>=</mo><mrow><mo>[</mo> <mrow>
  <mtable>
   <mtr>
    <mtd>
     <mrow>
      <mfrac>
       <mrow>
        <mi>δ</mi><msubsup>
         <mi>a</mi>
         <mn>1</mn>
         <mrow>
          <mrow><mo>(</mo>
           <mi>h</mi>
          <mo>)</mo></mrow>
         </mrow>
        </msubsup>
        <mrow><mo>(</mo>
         <mi>x</mi>
        <mo>)</mo></mrow>
       </mrow>
       <mrow>
        <mi>δ</mi><msub>
         <mi>x</mi>
         <mn>1</mn>
        </msub>
       </mrow>
      </mfrac>
     </mrow>
    </mtd>
    <mtd>
     <mo>⋯</mo>
    </mtd>
    <mtd>
     <mrow>
      <mfrac>
       <mrow>
        <mi>δ</mi><msubsup>
         <mi>a</mi>
         <mn>1</mn>
         <mrow>
          <mrow><mo>(</mo>
           <mi>h</mi>
          <mo>)</mo></mrow>
         </mrow>
        </msubsup>
        <mrow><mo>(</mo>
         <mi>x</mi>
        <mo>)</mo></mrow>
       </mrow>
       <mrow>
        <mi>δ</mi><msub>
         <mi>x</mi>
         <mi>m</mi>
        </msub>
       </mrow>
      </mfrac>
     </mrow>
    </mtd>
   </mtr>
   <mtr>
    <mtd>
     <mo>⋮</mo>
    </mtd>
    <mtd>
     <mo>⋱</mo>
    </mtd>
    <mtd>
     <mo>⋮</mo>
    </mtd>
   </mtr>
   <mtr>
    <mtd>
     <mrow>
      <mfrac>
       <mrow>
        <mi>δ</mi><msubsup>
         <mi>a</mi>
         <mi>n</mi>
         <mrow>
          <mrow><mo>(</mo>
           <mi>h</mi>
          <mo>)</mo></mrow>
         </mrow>
        </msubsup>
        <mrow><mo>(</mo>
         <mi>x</mi>
        <mo>)</mo></mrow>
       </mrow>
       <mrow>
        <mi>δ</mi><msub>
         <mi>x</mi>
         <mn>1</mn>
        </msub>
       </mrow>
      </mfrac>
     </mrow>
    </mtd>
    <mtd>
     <mo>⋯</mo>
    </mtd>
    <mtd>
     <mrow>
      <mfrac>
       <mrow>
        <mi>δ</mi><msubsup>
         <mi>a</mi>
         <mi>n</mi>
         <mrow>
          <mrow><mo>(</mo>
           <mi>h</mi>
          <mo>)</mo></mrow>
         </mrow>
        </msubsup>
        <mrow><mo>(</mo>
         <mi>x</mi>
        <mo>)</mo></mrow>
       </mrow>
       <mrow>
        <mi>δ</mi><msub>
         <mi>x</mi>
         <mi>m</mi>
        </msub>
       </mrow>
      </mfrac>
     </mrow>
    </mtd>
   </mtr>
  </mtable>
 </mrow> <mo>]</mo></mrow>
</math>

Written more succinctly, we can define our complete loss function as :-

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow class="MJX-TeXAtom-ORD">
    <mi class="MJX-tex-caligraphic" mathvariant="script">L</mi>
  </mrow>
  <mrow>
    <mo>(</mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>x</mi>
      <mo>,</mo>
      <mrow class="MJX-TeXAtom-ORD">
        <mover>
          <mi>x</mi>
          <mo stretchy="false">&#x005E;<!-- ^ --></mo>
        </mover>
      </mrow>
    </mrow>
    <mo>)</mo>
  </mrow>
  <mo>+</mo>
  <mi>&#x03BB;<!-- λ --></mi>
  <mrow class="MJX-TeXAtom-ORD">
    <munder>
      <mo movablelimits="false">&#x2211;<!-- ∑ --></mo>
      <mi>i</mi>
    </munder>
    <msup>
      <mrow class="MJX-TeXAtom-ORD">
        <mrow>
          <mo symmetric="true">&#x2016;</mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mrow class="MJX-TeXAtom-ORD">
              <msub>
                <mi mathvariant="normal">&#x2207;<!-- ∇ --></mi>
                <mi>x</mi>
              </msub>
            </mrow>
            <msubsup>
              <mi>a</mi>
              <mi>i</mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mrow>
                  <mo>(</mo>
                  <mi>h</mi>
                  <mo>)</mo>
                </mrow>
              </mrow>
            </msubsup>
            <mrow>
              <mo>(</mo>
              <mi>x</mi>
              <mo>)</mo>
            </mrow>
          </mrow>
          <mo symmetric="true">&#x2016;</mo>
        </mrow>
      </mrow>
      <mn>2</mn>
    </msup>
  </mrow>
</math>

where ```∇xa(h)i(x)``` defines the ```gradient field``` of our hidden layer activations with respect to the input x, summed over all i training examples.


## Summary

An autoencoder is a neural network architecture capable of discovering structure within data in order to develop a compressed representation of the input. Many different variants of the general autoencoder architecture exist with the goal of ensuring that the compressed representation represents meaningful attributes of the original data input; typically the biggest challenge when working with autoencoders is getting your model to actually learn a meaningful and generalizable latent space representation.

Because autoencoders learn how to compress the data based on attributes (ie. correlations between the input feature vector) discovered from data during training, these models are typically only capable of reconstructing data similar to the class of observations of which the model observed during training.