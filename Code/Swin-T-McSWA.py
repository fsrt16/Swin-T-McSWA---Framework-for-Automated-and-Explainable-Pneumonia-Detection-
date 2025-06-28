import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D,
    Activation, Add, LayerNormalization, MultiHeadAttention, Reshape, Lambda, multiply
)
from tensorflow.keras.applications import DenseNet169

class TBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(TBlock, self).__init__()

    def call(self, input_tensor):
        filters = int(input_tensor.shape[-1])

        x1 = Conv2D(filters, (3, 3), padding='same', dilation_rate=2)(input_tensor)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        x2 = Conv2D(filters, (3, 3), padding='same')(input_tensor)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)

        gap = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(filters // 8, activation='relu')(gap)
        dense2 = Dense(filters, activation='sigmoid')(dense1)

        channel_attn = Lambda(lambda x: tf.expand_dims(tf.expand_dims(x, axis=1), axis=1))(dense2)
        x3 = multiply([input_tensor, channel_attn])

        combined = Add()([x1, x2, x3])
        out = Activation('relu')(combined)
        return out

class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads=4, ff_dim=128, dropout_rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

    def call(self, x):
        x_norm1 = LayerNormalization()(x)
        attn_output = MultiHeadAttention(num_heads=self.num_heads, key_dim=x.shape[-1])(x_norm1, x_norm1)
        attn_output = Dropout(self.dropout_rate)(attn_output)
        x1 = Add()([x, attn_output])

        x_norm2 = LayerNormalization()(x1)
        ff_output = Dense(self.ff_dim, activation='relu')(x_norm2)
        ff_output = Dense(x.shape[-1])(ff_output)
        ff_output = Dropout(self.dropout_rate)(ff_output)

        return Add()([x1, ff_output])

class NovelHybridTransformer(tf.keras.Model):
    def __init__(self, input_shape=(224, 224, 3), final_class=5):
        super(NovelHybridTransformer, self).__init__()
        self.input_layer = Input(shape=input_shape)

        self.backbone = DenseNet169(input_shape=input_shape, include_top=False, weights='imagenet')
        self.backbone.trainable = False

        self.t_block = TBlock()
        self.conv_reduction = Conv2D(64, kernel_size=(1, 1), padding='same')
        self.transformer1 = TransformerEncoderBlock(num_heads=4, ff_dim=128)
        self.transformer2 = TransformerEncoderBlock(num_heads=4, ff_dim=128)

        self.pool = GlobalAveragePooling2D()
        self.dense1 = Dense(128, activation='relu')
        self.drop1 = Dropout(0.5)
        self.dense2 = Dense(64, activation='relu')
        self.drop2 = Dropout(0.25)
        self.output_layer = Dense(final_class, activation='softmax')

    def call(self, inputs):
        x = self.backbone(inputs)
        x = self.t_block(x)

        x = self.conv_reduction(x)
        b, h, w, c = x.shape
        x = Reshape((h * w, c))(x)

        x = self.transformer1(x)
        x = self.transformer2(x)

        x = Reshape((h, w, c))(x)
        x = self.pool(x)

        x = self.dense1(x)
        x = self.drop1(x)
        x = self.dense2(x)
        x = self.drop2(x)
        return self.output_layer(x)


