# Global Pooling in Conv2d
In this we will compare the use of `GlobalPooling2d` and `MaxPooling2d` in `nn.Conv2d`.

## Pooling
There are various operations other than just `Global` and `MaxPooling` like
- GlobalMaxPooling2d
- GlobalAveragePooling2d

## Classical Conv2d
There are many classical `Conv2d` model architectures like `AlexNet`, `LeNet`. They are a combination of 
`MLPs` and `Conv2d` layers.

<img src="https://blog.paperspace.com/content/images/2022/07/feature_extractor_classifier-1.png"/>

When looking at the `AlexNet` architecture, it makes sense as the `Conv2d` layers extract features and them pass them onto `Linear` layers to find correlation b/w `feature vectors` and `targets`. 

However, these `Linear` layer have the high chance to `overfit` data. Althought, `Dropout Regularization` was introduced to mitigate this issue but it still remains.

## Global Average Pooling
For Example, consider a 4 class classification task, `1x1 conv` will help downsampling the feature maps until 4 and `global pooling` will help create `4` element long vector which will be used by the `loss` in calculating gradients.

Now, imagine for this same task, we have `8` feature maps of size `(3, 3)`. We can utilize `1x1` Conv2d in order to down-sample the `8` feature maps of 4. One way to derive a `4 element vector` from these feature maps is to compute the `average` of all pixels in each feature map and return that as a `single` element. This is essentially what global average pooling entails.

## Global Max Pooling
Just like the scenario above where we would like to produce a `4 element vector` from `4 matrices`, in this case instead of taking the average value of all pixels in each feature map, we take the `maximum `value and return that as an individual element in the vector representation of interest.

-----------
We can now `Compare` the performance of `ConvNet` Module using `GlobalAveragePooling2d` and `GlobalMaxPooling2d`