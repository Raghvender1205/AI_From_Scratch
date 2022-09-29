# DeepSpeech Speech Recognition in PyTorch

A PyTorch based End-to-End Pipeline of Speech Recognition using DeepSpeech

https://arxiv.org/abs/1412.5567

## Model
DeepSpeech architecture but with slight modifications. It has 2 modules 
1. N Layers of `Convolutional Neural Networks` to learn audio features
2. `Bidirectional Recurrent Neural Net (BiRNN)` to use the learned ResCNN audio features.

<img src="https://assets-global.website-files.com/5fbd459f3b05914cf70496d7/5fdfa5d8f483a0eef6ef7cb1_BHOBfDVTcGCQKTtp.png"/>

`CNN` are used to extract features from audio `spectrograms`. We use `Residual CNN` which is `skip connections` as they have a `flatter` loss surface
which makes it easier for models to navigate `loss` and find a `lower minima`.

`RNN` are used for sequence modeling. RNN processes audio features and make a prediction for each frame while using `context` from previous features. 
We are using `BiRNN` as we want the `context` of not only the frame before each step but the frames after it as well. 
We use `GRU` as a variant of `RNN` as it requires less computational resources.