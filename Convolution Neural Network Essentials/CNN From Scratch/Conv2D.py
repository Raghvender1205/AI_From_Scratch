import numpy as np


class Conv2D:
    """
    A Convolution Layer using 3x3 filters
    """
    def __init__(self, num_filters):
        self.num_filters = num_filters

        # filters is 3D array with dimension -> (num_filters, 3, 3)
        # We divide it by 9 to reduce the variance of the initial values
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iter_regions(self, image):
        """
        Generates all possible 3x3 image regions using a 'valid' padding
        - image -> 2D Numpy Array
        """
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                img_region = image[i : (i + 3), j : (j + 3)]
                yield img_region, i, j
    

    def forward(self, x):
        """
        Forward Pass of the Conv2D Layer using x (input).
        Input:
            x -> 2D Input Array
        Return:
            3D Numpy array with dimensions (h, w, num_filters)
        """
        self.last_input_shape = x

        h, w = x.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for img_region, i, j in self.iter_regions(x):
            output[i, j] = np.sum(img_region * self.filters, axis=(1, 2))
        return output


    def backward(self, dL_d_out, lr):
        """
        Performs Backpropgation for the Conv2D Layer.
        """
        d_L_d_filters = np.zeros(self.filters.shape)
        
        for img_region, i, j in self.iter_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += dL_d_out[i, j, f] * img_region

        # Update Filters
        self.filters -= lr * d_L_d_filters

        # We aren't returning anything here since we use Conv3x3 as the first layer in our CNN.
        # Otherwise, we'd need to return the loss gradient for this layer's inputs, just like every
        # other layer in our CNN.
        return None
