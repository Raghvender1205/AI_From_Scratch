import os
import io
import imageio
import medmnist
import ipywidgets
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, utils, Sequential, Model
from keras.optimizer_experimental.adam import Adam
from keras.metrics import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy

# Constants
SEED = 42
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.keras.utils.set_random_seed(SEED)

# HyperParameters
DATASET_NAME = 'organmnist3d'
BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (28, 28, 28, 1)
NUM_CLASSES = 11

# Optimizer
LR = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 60

# Tublet Embedding
PATCH_SIZE = (8, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT Model Architecture
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8

# Dataset
def prepare_dataset(data_info: dict):
    """
    MedMNIST v2 Dataset 
    """
    data_path = tf.keras.utils.get_file(origin=data_info['url'], md5_hash=data_info['MD5'])
    
    with np.load(data_path) as data:
        # Videos
        train_videos = data['train_images']
        test_videos = data['test_images']
        valid_videos = data['val_images']

        # Gel labels
        train_labels = data['train_labels'].flatten()
        valid_labels = data['val_labels'].flatten()
        test_labels = data['test_labels'].flatten()

    return (
        (train_videos, train_labels),
        (valid_videos, valid_labels),
        (test_videos, test_labels)
    )


# Data Preprocess and DataLoader
@tf.function
def preprocess(frames: tf.Tensor, label: tf.Tensor):
    frames = tf.image.convert_image_dtype(
        frames[
            ..., tf.newaxis
        ], # new axis to help in processing of Conv3D
        tf.float32,
    )
    label = tf.cast(label, tf.float32)

    return frames, label

def prepare_dataloader(
    videos: np.ndarray, labels: np.ndarray, 
    loader_type: str = 'train', batch_size: int = BATCH_SIZE,
):
    dataset = tf.data.Dataset.from_tensor_slices((videos, labels))
    if loader_type == 'train':
        dataset = dataset.shuffle(BATCH_SIZE * 2)
    
    dataloader = (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    ) 

    return dataloader

class TubletEmbedding(layers.Layer):
    """
    Implement TubletEmbedding for ViViT
    """
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='VALID'
        )

        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patchesd = self.flatten(projected_patches)

        return flattened_patchesd

    
class PositionalEmbedding(layers.Layer):
    """
    A Positional Encoder which adds pos. information to the encoded video tokens
    """
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
    
    def build(self, input_shape):
        _, n_tokens, _ = input_shape
        self.pos_embedding = layers.Embedding(
            input_dim=n_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=n_tokens, delta=1)
    
    def call(self, encoded_tokens):
        # Encode positions and add it to the encoded tokens
        encoded_positions = self.pos_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions

        return encoded_tokens
    
# Video ViT Model Build
def videovit(tublet_embedder, positional_encoder, input_shape=INPUT_SHAPE,
          transformer_layers=NUM_LAYERS, num_heads=NUM_HEADS, 
          embed_dim=PROJECTION_DIM, layer_norm_eps=LAYER_NORM_EPS, n_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)
    # Create and Encode Patches
    patches = tublet_embedder(inputs)
    encoded_patches = positional_encoder(patches)

    # Transformer Block
    for _ in range(transformer_layers):
        # Layer Normalization and MultiHead Self Attention
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attn_out = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)
        # Skip Connection
        x2 = layers.Add()([attn_out, encoded_patches])
        # LayerNormalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=embed_dim, activation=tf.nn.gelu),
            ]
        )(x3)
        # Skip Connection
        encoded_patches = layers.Add()([x3, x2])

    # Layer Normalization and Global Average Pooling
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)
    # Classify outputs
    outputs = layers.Dense(units=n_classes, activation='softmax')(representation)
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Train Script
def train():
    model = videovit(
        tublet_embedder=TubletEmbedding(
            embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
        ),
        positional_encoder=PositionalEmbedding(embed_dim=PROJECTION_DIM),
    )

    optimizer = Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            SparseCategoricalAccuracy(name='accuracy'),
            SparseTopKCategoricalAccuracy(5, name='top-5-accuracy')
        ],
    )

    _ = model.fit(trainloader, epochs=EPOCHS, validation_data=validloader)
    _, accuracy, top_5_accuracy = model.evaluate(testloader)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return model

# Helper Function for Inference
def box_for_grid(image_widget, fit):
    """
    Create a VBox to hold image 
    """
    if fit is not None:
        fit_str = "'{}'".format(fit)
    else:
        fit_str = str(fit)

    h = ipywidgets.HTML(value="" + str(fit_str) + "")

    # Make the green box with the image widget inside it
    boxb = ipywidgets.widgets.Box()
    boxb.children = [image_widget]

    # Compose into a vertical box
    vb = ipywidgets.widgets.VBox()
    vb.layout.align_items = "center"
    vb.children = [h, boxb]
    
    return vb

if __name__ == '__main__':
    # Dataset
    info = medmnist.INFO[DATASET_NAME]
    prepared_dataset = prepare_dataset(info)
    (train_videos, train_labels) = prepared_dataset[0]
    (valid_videos, valid_labels) = prepared_dataset[1]
    (test_videos, test_labels) = prepared_dataset[2]

    # Dataloaders
    trainloader = prepare_dataloader(train_videos, train_labels, "train")
    validloader = prepare_dataloader(valid_videos, valid_labels, "valid")
    testloader = prepare_dataloader(test_videos, test_labels, "test")

    # Train
    model = train()
    print(model)

    # Inference
    NUM_SAMPLES_VIZ = 25
    testsamples, labels = next(iter(testloader))
    testsamples, labels = testsamples[:NUM_SAMPLES_VIZ], labels[:NUM_SAMPLES_VIZ]

    ground_truths = []
    preds = []
    videos = []

    for i, (testsample, label) in enumerate(zip(testsamples, labels)):
        # Generate gif
        with io.BytesIO() as gif:
            imageio.mimsave(gif, (testsample.numpy() *
                                255).astype("uint8"), "GIF", fps=5)
            videos.append(gif.getvalue())

        # Get model prediction
        output = model.predict(tf.expand_dims(testsample, axis=0))[0]
        pred = np.argmax(output, axis=0)

        ground_truths.append(label.numpy().astype("int"))
        preds.append(pred)

    boxes = []
    for i in range(NUM_SAMPLES_VIZ):
        ib = ipywidgets.widgets.Image(value=videos[i], width=100, height=100)
        true_class = info["label"][str(ground_truths[i])]
        pred_class = info["label"][str(preds[i])]
        caption = f"T: {true_class} | P: {pred_class}"

        boxes.append(box_for_grid(ib, caption))

    ipywidgets.widgets.GridBox(
        boxes, layout=ipywidgets.widgets.Layout(
            grid_template_columns="repeat(5, 200px)")
    )