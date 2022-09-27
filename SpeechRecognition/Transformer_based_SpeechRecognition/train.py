from ast import Call
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import Callback, LearningRateScheduler
from keras.losses import CategoricalCrossentropy
from keras.optimizer_v1 import Adam
from preprocess import val_ds, vectorizer, max_target_len, ds
from model import Transformer

# Callbacks
class DisplayOutputs(Callback):
    def __init__(self, batch, idx_to_token, target_start_token_idx=27, 
                    target_end_token_idx=28):
        self.batch = batch
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.idx_to_char = idx_to_token

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 != 0:
            return
        
        source = self.batch['source']
        target = self.batch['target']
        bs = tf.shape(source)[0]
        preds = self.model.generate(source, self.target_start_token_idx)
        preds = preds.numpy()

        for i in range(bs):
            target_next = "".join([self.idx_to_char[_] for _ in target[i, :]])
            prediction = ''
            for idx in preds[i, :]:
                prediction += self.idx_to_char[idx]
                if idx == self.target_end_token_idx:
                    break
            print(f"Target: {target_next.replace('-', '')}")
            print(f"Prediction: {prediction}\n")

# Custom LearningRateScheduler
class CustomSchedule(LearningRateScheduler):
    def __init__(self, init_lr=0.00001, lr_after_warmup=0.001, final_lr=0.00001,
                    warmup_epochs=15, decay_epochs=85, steps_per_epoch=203):
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def calculate_lr(self, epoch):
        warmup_lr = (self.init_lr + 
                        ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch)
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.lr_after_warmup - (epoch - self.warmup_epochs)
            * (self.lr_after_warmup - self.final_lr) / (self.decay_epochs),)
        
        return tf.math.minimum(warmup_lr, decay_lr)
    
    def __call__(self, step):
        epoch = step // self.steps_per_epoch

        return self.calculate_lr(epoch)
    
# Train Model
batch = next(iter(val_ds))

# Vocab to convert predicted idx into char.
idx_to_char = vectorizer.get_vocabulary()
display_cb = DisplayOutputs(
    batch, idx_to_char, target_start_token_idx=2, target_end_token_idx=3
)

model = Transformer(
    num_hid=200,
    num_head=2,
    num_feed_forward=400,
    target_maxlen=max_target_len,
    num_layers_enc=4,
    num_layers_dec=1,
    num_classes=34,
)

loss_fn = CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
lr = CustomSchedule(
    init_lr=0.00001, lr_after_warmup=0.001, final_lr=0.00001, warmup_epochs=15,
    decay_epochs=85, steps_per_epoch=len(ds),
)
optimizer = Adam(lr)
model.compile(optimizer=optimizer, loss=loss_fn)

history = model.fit(ds, validation_data=val_ds, callbacks=[display_cb], epochs=1)