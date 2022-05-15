from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
import tensorflow as tf
import numpy as np
# from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Lambda, Reshape, LSTM, Flatten, Dropout,Conv1D
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Add, Concatenate, Multiply, Activation, MaxPooling2D, AveragePooling2D
from tensorflow.keras.utils import plot_model

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog, Model

import tensorflow.keras.backend as K
class MyRNNUAVClass_old(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        """Define the layers of a custom model.

        Arguments:
            input_dict (dict): Dictionary of input tensors, including "obs",
                "prev_action", "prev_reward", "is_training".
            num_outputs (int): Output tensor must be of size
                [BATCH_SIZE, num_outputs].
            options (dict): Model options.

        Returns:
            (outputs, feature_layer): Tensors of size [BATCH_SIZE, num_outputs]
                and [BATCH_SIZE, desired_feature_size].

        When using dict or tuple observation spaces, you can access
        the nested sub-observation batches here as well:

        Examples:
            >>> print(input_dict)
            {'prev_actions': <tf.Tensor shape=(?,) dtype=int64>,
             'prev_rewards': <tf.Tensor shape=(?,) dtype=float32>,
             'is_training': <tf.Tensor shape=(), dtype=bool>,
             'obs': OrderedDict([
                ('sensors', OrderedDict([
                    ('front_cam', [
                        <tf.Tensor shape=(?, 10, 10, 3) dtype=float32>,
                        <tf.Tensor shape=(?, 10, 10, 3) dtype=float32>]),
                    ('position', <tf.Tensor shape=(?, 3) dtype=float32>),
                    ('velocity', <tf.Tensor shape=(?, 3) dtype=float32>)]))])}
        """

        layer1 = slim.fully_connected(input_dict["obs"], 64, ...)
        layer2 = slim.fully_connected(layer1, 64, ...)
        ...
        return layerN, layerN_minus_1

    def value_function(self):
        """Builds the value function output.

        This method can be overridden to customize the implementation of the
        value function (e.g., not sharing hidden layers).

        Returns:
            Tensor of size [BATCH_SIZE] for the value function.
        """
        return tf.reshape(
            linear(self.last_layer, 1, "value", normc_initializer(1.0)), [-1])

    def custom_loss(self, policy_loss, loss_inputs):
        """Override to customize the loss function used to optimize this model.

        This can be used to incorporate self-supervised losses (by defining
        a loss over existing input and output tensors of this model), and
        supervised losses (by defining losses over a variable-sharing copy of
        this model's layers).

        You can find an runnable example in examples/custom_loss.py.

        Arguments:
            policy_loss (Tensor): scalar policy loss from the policy.
            loss_inputs (dict): map of input placeholders for rollout data.

        Returns:
            Scalar tensor for the customized loss for this model.
        """
        return policy_loss

    def custom_stats(self):
        """Override to return custom metrics from your model.

        The stats will be reported as part of the learner stats, i.e.,
            info:
                learner:
                    model:
                        key1: metric1
                        key2: metric2

        Returns:
            Dict of string keys to scalar tensors.
        """
        return {}
    
class MyRNNUAVClass_old(RecurrentNetwork):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyRNNUAVClass_old, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        cell_size = 256
    
        # Define input layers
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]))
        state_in_h = tf.keras.layers.Input(shape=(256, ))
        state_in_c = tf.keras.layers.Input(shape=(256, ))
        seq_in = tf.keras.layers.Input(shape=(), dtype=tf.int32)
    
        # Send to LSTM cell
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True,
            name="lstm")(
                inputs=input_layer,
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h, state_in_c])
        output_layer = tf.keras.layers.Dense(units=16, activation='relu')(lstm_out)
    
        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[output_layer, state_h, state_c])
        self.rnn_model.summary()
        

class MyRNNConv1DModel_v3(RecurrentNetwork):
    """
    Not yet implemented !!!!!
    下記情報を元にして、最新の rllib version に合うように一部修正。
    とにかく「動く版」を作成するのが目的
        Information: https://github.com/ray-project/ray/issues/6928

        This was (given that you are not using tf-eager) a problem in your model.
        Here is a working version of it (see code below).
        The trick was to correctly fold the time-rank into the batch rank before pushing it through the CNN,
        then correctly unfolding it again before the LSTM pass.

        For the eager case, there is actually a bug in RLlib, which I'll fix now (issue #6732).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyRNNConv1DModel_v3, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        cnn_shape = obs_space.shape
        self.cell_size = 132

        visual_size = cnn_shape[0] * cnn_shape[1]
        state_in_h = Input(shape=(self.cell_size,), name='h')
        state_in_c = Input(shape=(self.cell_size,), name='c')
        seq_in = Input(shape=(), name='seq_in', dtype=tf.int32)

        inputs = Input(shape=(None, visual_size), name='visual_inputs')  # Add time dim
        input_visual = inputs
        input_visual = tf.reshape(input_visual, [-1, cnn_shape[0], cnn_shape[1]])

        layer_1 = Conv1D(filters=64, kernel_size=5, strides=1, activation='relu', padding='valid') \
            (input_visual)
        layer_bn = BatchNormalization()(layer_1)
        layer_2 = Conv1D(filters=128, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (layer_bn)
        layer_3 = Conv1D(filters=128, kernel_size=5, strides=3, activation='relu', padding='valid') \
            (layer_2)
        layer_4 = Conv1D(filters=128, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (layer_3)
        # vision_out = Flatten()(layer_3)
        vision_out = Lambda(lambda x: tf.squeeze(x, axis=1))(layer_4)

        vision_out = tf.reshape(vision_out, [-1, tf.shape(inputs)[1], vision_out.shape.as_list()[-1]])

        lstm_out, state_h, state_c = LSTM(units=self.cell_size,
                                          activation='tanh',
                                          recurrent_activation='sigmoid',
                                          return_state=True,
                                          return_sequences=True,
                                          name='lstm')(inputs=vision_out,
                                                       mask=tf.sequence_mask(lengths=seq_in),
                                                       initial_state=[state_in_h, state_in_c])
        layer_5 = Dense(units=64, activation='relu')(lstm_out)
        layer_6 = Dropout(rate=0.3)(layer_5)
        layer_7 = Dense(units=32, activation='relu')(layer_6)
        logits = Dense(units=num_outputs, activation=None, name='logits')(layer_7)

        val_5 = Dense(units=64, activation='relu')(lstm_out)
        val_6 = Dropout(rate=0.3)(val_5)
        val_7 = Dense(units=32, activation='relu')(val_6)
        values = Dense(units=1, activation=None, name='values')(val_7)

        self.rnn_model = Model(inputs=[inputs, seq_in, state_in_h, state_in_c],
                               outputs=[logits, values, state_h, state_c])

        # self.rnn_model.summary()
        # plot_model(self.rnn_model, to_file='rnn_model.png', show_shapes=True)

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        # inputs:(None, None, 30),  state=[(None,128),(None,128)], seq_len:(None,)
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] + state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        return [np.zeros(self.cell_size, np.float32), np.zeros(self.cell_size, np.float32)]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])
    
class MyRNNConv1DModel_v3_Attention(RecurrentNetwork):
    """
    Not yet implemented !!!!!
    Batch_Normalization, Dropout, Block Convolutional Attention Module 追加晩
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyRNNConv1DModel_v3, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        # cnn_shape = (10, 3)  # temporal setting
        cnn_shape = obs_space.shape
        self.cell_size = 128

        visual_size = cnn_shape[0] * cnn_shape[1]
        state_in_h = Input(shape=(self.cell_size,), name='h')
        state_in_c = Input(shape=(self.cell_size,), name='c')
        seq_in = Input(shape=(), name='seq_in', dtype=tf.int32)

        inputs = Input(shape=(None, visual_size), name='visual_inputs')  # Add time dim
        input_visual = inputs
        input_visual = tf.reshape(input_visual, [-1, cnn_shape[0], cnn_shape[1]])

        layer_1 = Conv1D(filters=64, kernel_size=5, strides=1, activation='relu', padding='valid') \
            (input_visual)
        channel_attention_1 = ChannelAttentionModule(layer_1)
        spatial_attention_1 = SpatialAttentionModule(channel_attention_1, kernel_size=46)
        res_1 = Add()([layer_1, spatial_attention_1])

        layer_bn = BatchNormalization()(res_1)
        layer_2 = Conv1D(filters=128, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (layer_bn)
        channel_attention_2 = ChannelAttentionModule(layer_2)
        spatial_attention_2 = SpatialAttentionModule(channel_attention_2, kernel_size=21)
        res_2 = Add()([layer_2, spatial_attention_2])

        layer_3 = Conv1D(filters=128, kernel_size=5, strides=3, activation='relu', padding='valid') \
            (res_2)
        channel_attention_3 = ChannelAttentionModule(layer_3)
        spatial_attention_3 = SpatialAttentionModule(channel_attention_3, kernel_size=6)
        res_3 = Add()([layer_3, spatial_attention_3])

        layer_4 = Conv1D(filters=128, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (res_3)
        # vision_out = Flatten()(layer_3)
        vision_out = Lambda(lambda x: tf.squeeze(x, axis=1))(layer_4)

        vision_out = tf.reshape(vision_out, [-1, tf.shape(inputs)[1], vision_out.shape.as_list()[-1]])

        lstm_out, state_h, state_c = LSTM(units=self.cell_size,
                                          activation='tanh',
                                          recurrent_activation='sigmoid',
                                          return_state=True,
                                          return_sequences=True,
                                          name='lstm')(inputs=vision_out,
                                                       mask=tf.sequence_mask(lengths=seq_in),
                                                       initial_state=[state_in_h, state_in_c])
        layer_5 = Dense(units=64, activation='relu')(lstm_out)
        layer_6 = Dropout(rate=0.3)(layer_5)
        layer_7 = Dense(units=32, activation='relu')(layer_6)
        logits = Dense(units=num_outputs, activation=None, name='logits')(layer_7)

        val_5 = Dense(units=64, activation='relu')(lstm_out)
        val_6 = Dropout(rate=0.3)(val_5)
        val_7 = Dense(units=32, activation='relu')(val_6)
        values = Dense(units=1, activation=None, name='values')(val_7)

        self.rnn_model = Model(inputs=[inputs, seq_in, state_in_h, state_in_c],
                               outputs=[logits, values, state_h, state_c])

        # self.rnn_model.summary()
        # plot_model(self.rnn_model, to_file='rnn_model.png', show_shapes=True)

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        # inputs:(None, None, 30),  state=[(None,128),(None,128)], seq_len:(None,)
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] + state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        return [np.zeros(self.cell_size, np.float32), np.zeros(self.cell_size, np.float32)]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])

def ChannelAttentionModule(input: tf.keras.Model, ratio=8):
    """
    CBAM: Convolutional Block Attention Module
    Define Channel Attention Modules for 1D image
    ref. https://cocoinit23.com/keras-channel-attention-spatial-attention/
    """
    channel = input.shape[-1]
    shared_dense_1 = Dense(units=channel // ratio,
                           activation='relu',
                           kernel_initializer='he_normal',
                           use_bias=True,
                           bias_initializer='zeros')
    shared_dense_2 = Dense(units=channel,
                           activation=None,
                           kernel_initializer='he_normal',
                           use_bias=True,
                           bias_initializer='zeros')

    avg_pooling = GlobalAveragePooling2D()(input)
    avg_pooling = Reshape((1, 1, channel))(avg_pooling)
    avg_pooling = shared_dense_1(avg_pooling)
    avg_pooling = shared_dense_2(avg_pooling)

    max_pooling = GlobalMaxPooling2D()(input)
    max_pooling = Reshape((1, 1, channel))(max_pooling)
    max_pooling = shared_dense_1(max_pooling)
    max_pooling = shared_dense_2(max_pooling)

    x = Add()([avg_pooling, max_pooling])
    x = Activation('sigmoid')(x)

    out = Multiply()([input, x])
    return out

def SpatialAttentionModule(input: tf.keras.Model, kernel_size=7):
    """
    CBAM: Convolutional Block Attention Module
    Define Spatial Attention Modules for 1D image
    ref. https://cocoinit23.com/keras-channel-attention-spatial-attention/
    """
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(input)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(input)
    x = Concatenate(axis=3)([avg_pool, max_pool])

    x = Conv2D(filters=1,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
               activation='sigmoid',
               kernel_initializer='he_normal',
               use_bias=False)(x)

    out = Multiply()([input, x])
    return out
