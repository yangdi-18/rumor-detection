import tensorflow as tf
from keras import initializers, regularizers, constraints
from keras.initializers import initializers_v2
from keras.layers import *
import keras.backend as K
"""
    使用点积注意力对每一个时间步计算权重加权求和,
    得到最终模型输出
"""
#attention代码
class GlobalAttention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.features_dim = 0

        super(GlobalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        #  print(input_shape)
        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, training=False,att_mask=None, **kwargs):
        # batch_size_num_steps,feature_dims
        features_dim = self.features_dim
        step_dim = tf.shape(x)[1]
        batch_size = tf.shape(x)[0]
        e = K.reshape(K.dot(x, K.reshape(self.W, (features_dim, 1))),
                      (batch_size, step_dim))  # e = K.dot(x, self.W)#batch_size,num_step
        if self.bias:
            e += self.b
        e = K.tanh(e)
        if att_mask is not None:
            e += tf.cast(att_mask,e.dtype) * -1e7
        a = K.exp(e)
        # apply mask after the exp. will be re-normalized next
        # if mask is not None:
        # cast the mask to floatX to avoid float64 upcasting in theano
        #    a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())  # batch
        a = K.expand_dims(a)
        c = K.sum(a * x, axis=1)
        return c
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'W_regularizer':self.W_regularizer,
            'b_regularizer':self.b_regularizer,
            'W_constraint':self.W_constraint,
            'b_constraint':self.b_constraint,
            'bias':self.bias,
        })
        return config
    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

"""
    生成双向时间编码,后续用于生成相对位置编码
"""
def relative_positional_embedding(max_len, d_model):
    position_left_dims = tf.range(start=max_len - 1, limit=0, delta=-1)
    position_right_dims = tf.range(start=0, limit=max_len, delta=1)
    position_dims = tf.expand_dims(tf.concat([position_left_dims, position_right_dims], axis=0), axis=-1)
    embed_dims = tf.range(d_model)[tf.newaxis, :]
    angle_rates = 1 / tf.pow(
        10000.0, tf.cast(
            (2 * (embed_dims // 2)) / d_model, tf.float32))
    angle_rads = tf.cast(position_dims, tf.float32) * angle_rates

    sines = tf.sin(angle_rads[:, 0::2])  # shape=(seq_len,d_model//2)
    sines = tf.pad(tf.expand_dims(sines, axis=-1), [[0, 0], [0, 0], [0, 1]], constant_values=0.)
    sines = tf.reshape(sines, (tf.shape(sines)[0], tf.shape(sines)[1] * 2))
    cosines = tf.cos(angle_rads[:, 1::2])
    cosines = tf.pad(tf.expand_dims(cosines, axis=-1), [[0, 0], [0, 0], [1, 0]], constant_values=0.)
    cosines = tf.reshape(cosines, (tf.shape(cosines)[0], tf.shape(cosines)[1] * 2))

    pos_encoding = cosines + sines
    return tf.expand_dims(pos_encoding, axis=0)

"""
    对时间编码实现裁剪功能,裁剪成目标尺寸
"""
class RelativeSinusoidalPositionalEncoding(Layer):
    """
        Relative Sinusoidal Positional Encoding

        Positional encoding for left context (sin) and right context (cos)
        Total context = 2 * max_len - 1
    """

    def __init__(self, max_len, dim_model, *args, **kwargs):
        super(RelativeSinusoidalPositionalEncoding, self).__init__(*args, **kwargs)
        init = initializers_v2.Constant(tf.constant(relative_positional_embedding(max_len, dim_model)))
        self.pos_encoding = self.add_weight(shape=(1, max_len * 2 - 1, dim_model),
                                            initializer=init, trainable=False, name="pos_emb")
        self.max_len = max_len

    def call(self, seq_len=0, hidden_len=0, causal=True):
        # Causal Context
        if causal:

            # (B, Th + T, D)
            if seq_len != 0:
                R = self.pos_encoding[:, self.max_len - seq_len - hidden_len: self.max_len]

            # (B, Tmax, D)
            else:
                R = self.pos_encoding[:, :self.max_len, :]

        # Full Context
        else:

            # (B, Th + 2*T-1, D)
            if seq_len != 0:
                R = self.pos_encoding[:, self.max_len - seq_len - hidden_len: self.max_len - 1 + seq_len]

            # (B, 2*Tmax-1, D)
            else:
                R = self.pos_encoding[:, :]

        return R

"""
    实现基于相对位置编码的自注意力
    这里是transformer的核心.
"""
class SelfRelMultiHeadAttention(Layer):
    """Multi-Head Self-Attention Layer with Relative Sinusoidal Positional Encodings

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        causal: whether the attention is causal or unmasked
        max_pos_encoding: maximum relative distance between elements

    References:
        Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context, Dai et al.


    """

    def __init__(self, dim_model, num_heads, causal, rel_pos_enc, scope="rmha"):
        super(SelfRelMultiHeadAttention, self).__init__(name=scope)
        self.dim_model = dim_model
        # Position Embedding Layer
        self.scope = scope
        self.num_heads = num_heads
        self.dim_head = self.dim_model // self.num_heads
        self.pos_layer = Dense(self.dim_model, name="%s/pos_dense" % self.scope)
        self.query_layer = Dense(self.dim_model, name="%s/query_dense" % self.scope, use_bias=False)
        self.key_layer = Dense(self.dim_model, name="%s/key_dense" % self.scope, use_bias=False)
        self.value_layer = Dense(self.dim_model, name="%s/value_dense" % self.scope, use_bias=False)
        self.output_layer = Dense(self.dim_model, name="%s/out_dense" % self.scope, use_bias=False)
        self.causal = causal

        # Global content and positional bias
        self.u = self.add_weight(shape=(1, 1, self.dim_model), name="%s/param_u" % self.scope)  # Content bias
        self.v = self.add_weight(shape=(1, 1, self.dim_model), name="%s/param_v" % self.scope)  # Pos bias

        # Relative Sinusoidal Positional Encodings
        self.rel_pos_enc = rel_pos_enc

    def rel_to_abs(self, att_scores):

        """Relative to absolute position indexing

        Args:
            att_scores: absolute-by-relative indexed attention scores of shape
            (B, H, T, Th + 2*T-1) for full context and (B, H, T, Th + T) for causal context

        Return:
            att_scores: absolute-by-absolute indexed attention scores of shape (B, H, T, Th + T)

        References:
            causal context:
            Music Transformer, Huang et al.

            full context:
            Attention Augmented Convolutional Networks, Bello et al.


        """

        # Causal Context
        if self.causal:

            # Att Scores (B, H, T, Th + T)
            batch_size, num_heads, seq_length1, seq_length2 = \
                tf.shape(att_scores)[0], tf.shape(att_scores)[1], tf.shape(att_scores)[2], tf.shape(att_scores)[3]

            # Column Padding (B, H, T, 1 + Th + T)
            att_scores = tf.pad(att_scores, [[0, 0], [0, 0], [0, 0], [1, 0]], constant_values=0.)

            # Flatten (B, H, T + TTh + TT)
            att_scores = tf.reshape(att_scores, [batch_size, num_heads, -1])

            # Start Padding (B, H, Th + T + TTh + TT)
            att_scores = tf.pad(att_scores, [[0, 0], [0, 0], [seq_length2 - seq_length1, 0]], constant_values=0.)

            # Reshape (B, H, 1 + T, Th + T)
            att_scores = tf.reshape(att_scores, [batch_size, num_heads, 1 + seq_length1, seq_length2])

            # Slice (B, H, T, Th + T)
            att_scores = att_scores[:, :, 1:]

        # Full Context
        else:

            # Att Scores (B, H, T, Th + 2*T-1)
            batch_size, num_heads, seq_length1, seq_length2 = \
                tf.shape(att_scores)[0], tf.shape(att_scores)[1], tf.shape(att_scores)[2], tf.shape(att_scores)[3]

            # Column Padding (B, H, T, Th + 2*T)
            att_scores = tf.pad(att_scores, [[0, 0], [0, 0], [0, 0], [0, 1]], constant_values=0.)

            # Flatten (B, H, TTh + 2*TT)
            att_scores = tf.reshape(att_scores, [batch_size, num_heads, -1])

            # End Padding (B, H, TTh + 2*TT + Th + T - 1)
            att_scores = tf.pad(att_scores, [[0, 0], [0, 0], [0, seq_length2 - seq_length1]], constant_values=0.)

            # Reshape (B, H, T + 1, Th + 2*T-1)
            att_scores = tf.reshape(att_scores, [batch_size, num_heads, 1 + seq_length1, seq_length2])

            # Slice (B, H, T, Th + T)
            att_scores = att_scores[:, :, :seq_length1, seq_length1 - 1:]

        return att_scores

    def call(self, qkv, att_mask=None, training=False):

        """Scaled Dot-Product Self-Attention with relative sinusoidal position encodings

        Args:
            Q: Query of shape (B, T, D)
            K: Key of shape (B, T, D)
            V: Value of shape (B, T, D)
            mask: Optional position mask of shape (1 or B, 1 or H, 1 or T, 1 or T)
            hidden: Optional Key and Value hidden states for decoding

        Return:
            O: Attention output of shape (B, T, D)
            att_w: Attention weights of shape (B, H, T, Th + T)
            hidden: Key and value hidden states
            :param training: A

        """

        # Batch size B
        batch_size = tf.shape(qkv)[0]

        # Linear Layers
        Q = self.query_layer(qkv)
        K = self.key_layer(qkv)
        V = self.value_layer(qkv)  # bz,seq_len,d_model
        # Hidden State Provided
        # Update Hidden State

        # Add Bias
        Qu = Q + self.u
        Qv = Q + self.v

        # Relative Positional Embeddings (B, Th + 2*T-1, D) / (B, Th + T, D)
        E = self.pos_layer(self.rel_pos_enc(tf.shape(Q)[1], tf.shape(K)[1] - tf.shape(Q)[1], causal=self.causal))
        # print('E:',E.shape)
        E = tf.tile(E, [batch_size, 1, 1])
        # Reshape and Transpose (B, T, D) -> (B, H, T, d)
        Qu = tf.reshape(Qu, (batch_size, -1, self.num_heads, self.dim_head))
        Qu = tf.transpose(Qu, [0, 2, 1, 3])
        Qv = tf.reshape(Qv, [batch_size, -1, self.num_heads, self.dim_head])
        Qv = tf.transpose(Qv, [0, 2, 1, 3])

        # Reshape and Transpose (B, Th + T, D) -> (B, H, Th + T, d)
        K = tf.reshape(K, [batch_size, -1, self.num_heads, self.dim_head])
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.reshape(V, [batch_size, -1, self.num_heads, self.dim_head])
        V = tf.transpose(V, [0, 2, 1, 3])
        # Reshape and Transpose (B, Th + 2*T-1, D) -> (B, H, Th + 2*T-1, d) / (B, Th + T, D) -> (B, H, Th + T, d)
        E = tf.reshape(E, [batch_size, -1, self.num_heads, self.dim_head])
        E = tf.transpose(E, [0, 2, 1, 3])

        # att_scores (B, H, T, Th + T)

        att_scores_K = tf.matmul(Qu, K, transpose_b=True)  # Qu.matmul(K.transpose(2, 3))
        att_scores_E = self.rel_to_abs(tf.matmul(Qv, E, transpose_b=True))

        att_scores = (att_scores_K + att_scores_E) / tf.cast(tf.shape(K)[-1], att_scores_K.dtype) ** 0.5
        # Apply mask
        att_dtype = att_scores.dtype
        att_scores = tf.cast(att_scores, dtype="float32")
        if att_mask is not None:
            att_mask = tf.cast(att_mask, dtype=att_scores.dtype)
            att_scores += (att_mask * -1e7)

        # Att weights (B, H, T, Th + T)
        att_w = tf.nn.softmax(att_scores, axis=-1)
        att_w = tf.cast(att_w, att_dtype)
        # Att output (B, H, T, d)
        O = tf.matmul(att_w, V)

        # Transpose and Reshape (B, H, T, d) -> (B, T, D)
        O = tf.transpose(O, [0, 2, 1, 3])
        O = tf.reshape(O, [batch_size, -1, self.dim_model])

        # Output linear layer
        O = self.output_layer(O)
        return O, att_w
