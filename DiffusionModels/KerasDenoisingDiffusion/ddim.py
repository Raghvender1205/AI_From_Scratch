# Denoising Diffusion in TensorFlow
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers

# Data Constants
dataset_name = 'oxford_flowers102'
dataset_repetitions = 1
n_epochs = 1 # 50
image_size = 64

# Kernel Inception Distance
kid_img_size = 75
kid_diffusion_steps = 5
plot_diffusion_steps = 20

# Sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# Arcitecture
emb_dim = 32
emb_max_freq = 1000.0
widths = [32, 64, 96, 128]
block_depth = 2

# Optimization
BS = 64
ema = 0.999
LR = 1e-3
weight_decay = 1e-4

# Data Preprocess
def preprocess_image(data):
    # Center Crop
    h = tf.shape(data['image'])[0]
    w = tf.shape(data['image'])[1]
    crop_size = tf.minimum(h, w)
    image = tf.image.crop_to_bounding_box(data['image'], (h - crop_size) // 2,
                                    (w - crop_size) // 2, crop_size, crop_size) 
    # Resize and Clip for Downsampling 
    image = tf.image.resize(image, size=[image_size, image_size], antialias=True)

    return tf.clip_by_value(image / 255.0, 0.0, 1.0)

def prepare_dataset(split):
    # Validation Split is to be Shuffled, as Data order matters for KID Estimation
    return (
        tfds.load(dataset_name, split=split, shuffle_files=True)
        .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .repeat(dataset_repetitions)
        .shuffle(10 * BS)
        .batch(BS, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

"""
Kernel Inception Distance

An Image Quality Metric in which images are evaulated at the minimal possible resolution of Inception Net.
As the dataset is small, we go over the train and val splits multiple times per epoch as KID Estimation
is Noisy and compute expensive.
"""
class KID(keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        # KID is estimated per batch and is Averaged across batches
        self.kid_tracker = keras.metrics.Mean(name='kid_tracker')

        # Pretrained InceptionV3 is used without its classification Layer 
        # Some Preprocessing during PreTraining
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(image_size, image_size, 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=kid_img_size, width=kid_img_size),
                layers.Lambda(tf.keras.applications.inception_v3.preprocess_input),
                tf.keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(kid_img_size, kid_img_size, 3),
                    weights='imagenet',
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name='inception_encoder',
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dim = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)

        return (features_1 @ tf.transpose(features_2) / feature_dim + 1.0) ** 3.0
    
    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # Compute Polynomial Kernels using Two Feature Sets
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(generated_features, generated_features)
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # Estimate Max Mean Discrepancy using Average Kernel Values
        BS = tf.shape(real_features)[0]
        BS_f = tf.cast(BS, dtype=tf.float32)
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(BS))) / (BS_f * (BS_f - 1.0))
        mean_kernel_generated = tf.reduce_sum(kernel_generated * (1.0 - tf.eye(BS)) / (BS_f - 1.0))    
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated  - 2.0 * mean_kernel_cross

        # Update Average KID Estimate
        self.kid_tracker.update_state(kid)
    
    def result(self):
        return self.kid_tracker.result()
    
    def reset_state(self):
        self.kid_tracker.reset_state()


"""
Network Architecture

1. Sinusoidal Embedding
2. Residual Block
3. Down Block
4. Up Block
"""
def sinusoidal_embedding(x):
    emb_max_freq = 1.0
    freq = tf.exp(tf.linspace(
        tf.math.log(emb_max_freq),
        tf.math.log(emb_max_freq),
        emb_dim // 2,
    ))
    angular_speeds = 2.0 * math.pi * freq
    embeddings = tf.concat([tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3)

    return embeddings

def ResidualBlock(width):
    def apply(x: tf.Tensor):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(width, kernel_size=3, padding='same', activation=tf.keras.activations.swish)()
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply

def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply

def get_network(image_size, widths, block_depth):
    noisy_images = tf.keras.Input(shape=(image_size, image_size, 3))
    noise_variances = tf.keras.Input(shape=(1, 1, 1))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)
    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])
    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)
    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return tf.keras.Model([noisy_images, noise_variances], x, name="residual_unet")

"""
Diffusion Model
"""
class DiffusionModel(tf.keras.Model):
    def __init__(self, image_size, widths, block_depth):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.network = get_network(image_size, widths, block_depth)
        self.ema_network = tf.keras.models.clone_model(self.network)

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(name="kid")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(max_signal_rate)
        end_angle = tf.acos(min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images, image_size, image_size, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(BS, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(BS, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(BS, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(BS, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=BS, diffusion_steps=kid_diffusion_steps
        )
        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=plot_diffusion_steps,
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == '__main__':
    train_dataset = prepare_dataset("train[:80%]+validation[:80%]+test[:80%]")
    val_dataset = prepare_dataset("train[80%:]+validation[80%:]+test[80%:]")

    model = DiffusionModel(image_size, widths, block_depth)
    model.compile(
        optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=LR, weight_decay=weight_decay),
        loss=tf.keras.losses.mean_absolute_error,
    )
    ckpt_path = 'ckpts/diffusion_model'
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        monitor='val_kid',
        mode='min',
        save_best_only=True,
    )
    # Cal Mean and Variance of Train Dataset
    model.normalizer.adapt(train_dataset)

    # TRAIN
    model.fit(train_dataset, epochs=n_epochs, validation_data=val_dataset,
                callbacks=[
                    tf.keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
                    ckpt_callback,
                ],)

    # Inference
    model.load_weights(ckpt_path)
    model.plot_images()