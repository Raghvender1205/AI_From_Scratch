import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Permute, Dense, LayerNormalization, Embedding
import tensorflow_addons as tfa
from .layers import quick_gelu, Layer

class CLIPAttention(Layer):
    def __init__(self):
        super().__init__()
        self.emb_dim = 768
        self.num_heads = 12
        self.head_dim = self.emb_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = Dense(self.emb_dim)
        self.k_proj = Dense(self.emb_dim)
        self.v_proj = Dense(self.emb_dim)
        self.out_proj = Dense(self.emb_dim)

    def _shape(self, tensor, seq_len: int, batch_size: int):
        a = tf.reshape(tensor, (batch_size, seq_len, self.num_heads, self.head_dim))

        return Permute((2, 1, 3))(a) # bs, n_head, seq_len, head_dim

    def call(self, inputs):
        hidden_states, casual_attention_mask = inputs
        batch_size, tgt_len, emb_dim = hidden_states.shape
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), tgt_len, -1)
        value_states = self._shape(self.v_proj(hidden_states), tgt_len, -1)

        proj_shape = (-1, tgt_len, self.head_dim)
        query_states = self._shape(query_states, tgt_len, -1)
        query_states = tf.reshape(query_states, proj_shape)
        key_states = tf.reshape(key_states, proj_shape)

        src_len = tgt_len
        value_states = tf.reshape(value_states, proj_shape)
        attn_weights = query_states @ Permute((2, 1))(key_states)
        attn_weights = tf.reshape(attn_weights, (-1, self.num_heads, tgt_len, src_len))
        attn_weights = attn_weights + casual_attention_mask
        attn_weights = tf.reshape(attn_weights, (-1, tgt_len, src_len))

        attn_weights = tf.nn.softmax(attn_weights)
        attn_output = attn_weights @ value_states
        attn_output = tf.reshape(attn_output, (-1, self.num_heads, tgt_len, self.head_dim))
        attn_output = Permute((2, 1, 3))(attn_output)
        attn_output = tf.reshape(attn_output, (-1, tgt_len, emb_dim))

        return self.out_proj(attn_output)
    

class CLIPEncoderLayer(Layer):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = LayerNormalization(epsilon=1e-5)
        self.self_attn = CLIPAttention()
        self.layer_norm2 = LayerNormalization(epsilon=1e-5)
        self.fc1 = Dense(3072)
        self.fc2 = Dense(768)
    
    def call(self, inputs):
        hidden_states, casual_attention_mask = inputs
        residual = hidden_states
        
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn([hidden_states, casual_attention_mask])
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = quick_gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)

        return residual + hidden_states

class CLIPEncoder(Layer):
    def __init__(self):
        super().__init__()
        self.layers = [CLIPEncoderLayer() for _ in range(12)]

    def call(self, x):
        [hidden_states, casual_attention_mask] = x
        for layer in self.layers:
            hidden_states = layer([hidden_states, casual_attention_mask])
        return hidden_states


class CLIPTextEmbedding(Layer):
    def __init__(self, n_words=77):
        super().__init__()
        # Token and Position Embedding Layer
        self.token_embedding = Embedding(
            49408, 768, name="token_embedding"
        )
        self.position_embedding = Embedding(
            n_words, 768, name="position_embedding"
        )

    def call(self, x):
        input_ids, pos_ids = x
        word_embeddings = self.token_embedding(input_ids)
        pos_embeddings = self.position_embedding(pos_ids)

        return word_embeddings + pos_embeddings

class CLIPTextTransformer(Model):
    def __init__(self, n_words=77):
        super().__init__()
        self.embeddings = CLIPTextEmbedding(n_words=n_words)
        self.encoder = CLIPEncoder()
        self.final_layer_norm = LayerNormalization(epsilon=1e-5)
        self.casual_attention_mask = tf.constant(
            np.triu(np.ones((1, 1, 77, 77), dtype='float32') * -np.inf, k=1)
        )

    def call(self, inputs):
        input_ids, pos_ids = inputs
        x = self.embeddings([input_ids, pos_ids])
        x = self.encoder([x, self.casual_attention_mask])

        return self.final_layer_norm(x)
        