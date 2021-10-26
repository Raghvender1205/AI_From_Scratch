import numpy as np


class MaxPooling2D:
    # Max Pooling Layer using a pool size of 2.
    def iter_regions(self, image):
        """
        Generates a non-overlapping 2x2 image regions to pool over.
        - A 2D Numpy Array
        """
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                img_region = image[(i * 2) : (i * 2 + 2), (j, 2) : (j * 2 + 2)]
                yield img_region, i, j

    def forward(self, x): # x => Input
        """
        Forward Pass for the maxpooling layer using x (Input).

        Args:
            3D Numpy Array with dims (h, w, num_filters)
        Return:
            A 3D Numpy array with dims -> (h/2, w/2, num_filters).
        """
        self.last_input = x

        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for img_region, i, j in self.iter_regions(x):
            output[i, j] = np.amax(img_region, axis=(0, 1))

        return output

    # Backpropogation -> (dL / d_out)
    def backward(self, dL_d_out): 
        """
        Performs a backward pass of the maxpooling layer.

        Return:
            The loss gradient for the layer's inputs.
            - dL_d_out -> Loss gradient for this layer's outputs
        """
        dL_d_input = np.zeros(self.last_input.shape)

        for img_region, i, j in self.iter_regions(self.last_input):
            h, w, f = img_region.shape
            amax = np.amax(img_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If the pixel has the max_value, then copy the gradient to it.
                        if img_region[i2, j2, f2] == amax[f2]:
                            dL_d_input[i * 2 + i2, j * 2 + j2, f2] = dL_d_out[i, j, f2]

        return dL_d_input