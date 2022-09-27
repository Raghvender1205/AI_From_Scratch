import os
import random
from glob import glob
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv1D, Embedding, MultiHeadAttention, Dense
from keras.layers import LayerNormalization, Dropout, Layer
from keras import Sequential, Model
from keras.metrics import Mean

# Input Layer
class TokenEmbedding(Layer):
    def __init__(self, num_vocab=1000, maxlen=10, num_hid=64):
        super().__init__()
        self.emb = Embedding(num_vocab, num_hid)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        pos = tf.range(start=0, limit=maxlen, delta=1)
        pos = self.pos_emb(pos)
        
        return x + pos


class SpeechFeatureEmbedding(Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = Conv1D(num_hid, 11, strides=2, padding='same', activation='relu')
        self.conv2 = Conv1D(num_hid, 11, strides=2, padding='same', activation='relu')
        self.conv3 = Conv1D(num_hid, 11, strides=2, padding='same', activation='relu')
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return self.conv3(x)
    

# Encoder Layer
class Encoder(Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(feed_forward_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.attn(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)


# Decoder Layer
class Decoder(Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = LayerNormalization(epsilon=1e-6)        
        self.layernorm2 = LayerNormalization(epsilon=1e-6)        
        self.layernorm3 = LayerNormalization(epsilon=1e-6)        
        self.self_attn = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = Dropout(0.5)
        self.enc_dropout = Dropout(0.1)
        self.ffn_dropout = Dropout(0.1)
        self.ffn = Sequential([
            Dense(feed_forward_dim, activation='relu'),
            Dense(embed_dim)
        ])
    
    def casual_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """
        It masks the upper half of the dot product in Self Attention

        This prevents flow of information from future tokens to current token. 1's 
        in the lower triangle, counting from the lowest right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )

        return tf.tile(mask, mult)

    def call(self, enc_out, target):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        casual_mask = self.casual_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_attn = self.self_attn(target, target, attention_mask=casual_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_attn))
        
        enc_out = self.enc_attn(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))

        return ffn_out_norm


class Transformer(Model):
    def __init__(self, num_hid=64, num_head=2, num_feed_forward=128, src_maxlen=100,
                 target_maxlen=100, num_layers_enc=4, num_layers_dec=1, num_classes=10):
        super().__init__()
        self.loss_metric = Mean(name='loss')
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        self.enc_input = SpeechFeatureEmbedding(
            num_hid=num_hid, maxlen=src_maxlen)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )

        self.encoder = Sequential(
            [self.enc_input]
            + [
                Encoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ]
        )
        for i in range(num_layers_dec):
            setattr(self, f'dec_layer_{i}', Decoder(
                num_hid, num_head, num_feed_forward))

        self.classifier = Dense(num_classes)

    def decode(self, enc_out, target):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f'dec_layer_{i}')(enc_out, y)

        return y

    def call(self, inputs):
        src = inputs[0]
        target = inputs[1]
        x = self.encoder(src)
        y = self.decode(x, target)

        return self.classifier(y)

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch):
        source = batch['source']
        target = batch['target']
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]

        with tf.GradientTape() as tape:
            preds = self([source, dec_input])
            one_hot = tf.one_hot(dec_input, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        trainble_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainble_vars)
        self.optimizer.apply_gradients(zip(gradients, trainble_vars))
        self.loss_metric.update_state(loss)

        return {'loss': self.loss_metric.result()}

    def test_step(self, batch):
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input])
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)

        return {"loss": self.loss_metric.result()}

    def generate(self, src, target_start_token_idx):
        """
        Performs Inference over on batch of inputs using greedy decoding 
        """
        bs = tf.shape(src)[0]
        enc = self.encoder(src)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        dec_logits = []

        for i in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input)
            logits = self.classifier(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        
        return dec_input