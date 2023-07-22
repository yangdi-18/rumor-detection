import argparse

import keras.backend as K
import tensorflow as tf
from keras import Sequential
from keras import initializers, regularizers, constraints
from keras.layers import Layer, Bidirectional, Dense, Dropout, LayerNormalization, InputLayer, \
    Embedding, GRU
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.optimizers.schedules.learning_rate_schedule import LearningRateSchedule
from keras.saving.save import save_model
from sklearn.model_selection import train_test_split

from data_loader import get_2class_emotion_dataset, get_6class_emotion_dataset
from shared_blocks import GlobalAttention

#tf.config.set_visible_devices(tf.config.get_visible_devices()[:1] + tf.config.get_visible_devices()[2:])
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
#tf.compat.v1.Session(config=config)
"""
    热启动余弦衰减学习率函数
"""
class CosineDecayWarmup(LearningRateSchedule):

    def __init__(self, init_lr, steps, warmup_steps, min_lr):
        super(CosineDecayWarmup, self).__init__()

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.cosine_decay = tf.keras.experimental.CosineDecay(
            init_lr, steps - warmup_steps, min_lr)

    def __call__(self, step):
        linear_increase = self.init_lr * tf.cast(step, tf.float32) / (
                tf.cast(self.warmup_steps, tf.float32) + 1e-5)
        cosine_decay = self.cosine_decay(step)
        return tf.cond(pred=step <= self.warmup_steps,
                       true_fn=lambda: linear_increase,
                       false_fn=lambda: cosine_decay)

    def get_config(self):
        return {
            'warmup_steps': self.warmup_steps,
            'init_lr': self.init_lr
        }




MIN_SEQ_LEN = 16
MAX_SEQ_LEN = 96
RNN_UNITS = 384
ATT_HEADS = 4
ATT_UNITS = 128
VOCAB_SIZE = 21128
DROP_RATE = 0.2
NUM_CLASSES = 2

"""
    BiLSTM的基本层
    每一层有两个双向GRU+一个多头注意力层构成
    
"""
class BiRnnWithAttention(Layer):
    def __init__(self, rnn_units, att_units, att_heads, dropout, cell_type='lstm', *args, name='rnn_attention',
                 **kwargs):
        super(BiRnnWithAttention, self).__init__(*args, name=name, **kwargs)
        self.rnn_units = rnn_units
        self.att_units = att_units
        self.att_heads = att_heads
        self.dropout = dropout
        self.cell_type = cell_type
        self.layer_name = name
        self.rnn1 = Bidirectional(GRU(rnn_units, name='rnn1', return_sequences=True))
        self.rnn2 = Bidirectional(GRU(rnn_units, name='rnn2', return_sequences=True))
        self.drop1 = Dropout(dropout, name='drop1')
        self.drop2 = Dropout(dropout, name='drop2')
        self.drop3 = Dropout(dropout, name='drop3')
        self.drop4 = Dropout(dropout, name='drop4')
        self.ln1 = LayerNormalization(epsilon=1e-7, name='ln1')
        self.ln2 = LayerNormalization(epsilon=1e-7, name='ln2')
        self.concat_dense = Dense(rnn_units, activation='gelu', use_bias=False,name='concat')
        # self.ffn1_dense = Dense(rnn_units *3,activation='gelu',use_bias=False)
        # self.ffn2_dense = Dense(rnn_units,use_bias=False)
        self.attention = CustomMultiHeadAttention(att_units, rnn_units, rnn_units, att_heads, dropout,name='attention')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'rnn_units': self.rnn_units,
            'att_units': self.att_units,
            'att_heads': self.att_heads,
            'dropout': self.dropout,
            'cell_type': self.cell_type,
            'name': self.layer_name
        })
        return config

    def call(self, inputs, training=False, **kwargs):
        inputs = self.drop1(inputs, training=training)
        x_rnn = self.rnn1(inputs)
        x_rnn = self.drop2(x_rnn, training=training)
        x_rnn = self.concat_dense(x_rnn)
        x_rnn = self.rnn2(x_rnn)
        x_rnn = self.drop3(x_rnn, training=training)
        x_rnn = self.concat_dense(x_rnn)
        x = self.ln1(x_rnn + inputs)
        # print('A',x.shape)
        # x_ffn = self.ffn1_dense(x)
        # x_ffn = self.ffn2_dense(x_ffn)
        # x = x + x_ffn
        x_att = self.attention(x, x, x, training=training)
        x_att = self.drop4(x_att, training=training)
        o = self.ln2(x_att + x)
        return o

    def build(self, input_shape):
        return input_shape

"""
    构建情感分类模型的backbone
    因为我们需要获取分类模型的隐藏层输出 因此我们构建一个backbone模型 最后与一个输出层拼接得到训练模型
    在保存的时候 只保存backbone部分即可
"""
def build_backbone_model():
    model = Sequential([
        InputLayer(input_shape=(MAX_SEQ_LEN,), dtype=tf.int32),
        Embedding(VOCAB_SIZE, RNN_UNITS),
        BiRnnWithAttention(RNN_UNITS, ATT_UNITS, ATT_HEADS, DROP_RATE, name='bilstm1'),
        BiRnnWithAttention(RNN_UNITS, ATT_UNITS, ATT_HEADS, DROP_RATE, name='bilstm2'),
        BiRnnWithAttention(RNN_UNITS, ATT_UNITS, ATT_HEADS, DROP_RATE, name='bilstm3'),
        BiRnnWithAttention(RNN_UNITS, ATT_UNITS, ATT_HEADS, DROP_RATE, name='bilstm4'),
        GlobalAttention(),
        # Dense(NUM_CLASSES)
    ],name='emotion_backbone')
    return model


"""
class BiRnnWithAttention(Layer):
    def __init__(self,rnn_units,cell_type='lstm',*args,name='rnn_attention',**kwargs):
        super(BiRnnWithAttention,self).__init__(*args,name=name,**kwargs)
        self.cell = LSTMCell(rnn_units,name='')
        self.dense_cat = Dense(rnn_units,use_bias=False,activation='relu')

        pass
    def __get_init_state(self):
        hidden_states =
    def call(self, inputs, training=False):
        bz,seq_len,dim = inputs.shape
        for forward in ['left2right','right2left']:
            cell_states = self.__get_init_state()
            for step_input in tf.unstack(inputs,axis=1):
                cell_output,cell_states = self.cell(step_input,cell_states)
"""

"""
    点积注意力部分
"""
def scaled_dot_product_attention(k, q, v, mask):
    dk = tf.cast(tf.shape(k)[-1], 'float32')
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    scaled_matmul_qk = matmul_qk / tf.sqrt(dk)
    if mask is not None:
        masked_scaled_matmul_kq = scaled_matmul_qk + (-1e7) * mask
        attention_weights = tf.nn.softmax(masked_scaled_matmul_kq, axis=-1)
    else:
        attention_weights = tf.nn.softmax(scaled_matmul_qk, axis=-1)  # bn,head,q_len,k_len
    outputs = tf.matmul(attention_weights, v)  # bn,k_len,d_v
    return outputs, attention_weights

"""
    多头注意力部分
"""
class CustomMultiHeadAttention(Layer):
    def __init__(self, qk_units, v_units, out_units, num_heads, dropout=0.2,*args,**kwargs):
        super(CustomMultiHeadAttention, self).__init__(*args,**kwargs)
        self.num_heads = num_heads
        self.qk_units = qk_units
        self.v_units = v_units
        self.out_units = out_units
        self.qk_depth = self.qk_units // self.num_heads
        assert self.qk_depth * self.num_heads == self.qk_units
        self.v_depth = self.v_units // self.num_heads
        assert self.v_depth * self.num_heads == self.v_units
        self.dropout = dropout

        self.dense_q = Dense(self.qk_units, use_bias=False)
        self.dense_k = Dense(self.qk_units, use_bias=False)
        self.dense_v = Dense(self.v_units, use_bias=False)
        self.dense_o = Dense(self.out_units)
        self.drop = Dropout(self.dropout)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'qk_units': self.qk_units,
            'v_units': self.v_units,
            'out_units': self.out_units,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
        })
        return config

    def split_heads(self, x, batch_size, seq_len, depth):
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, depth))
        o = tf.transpose(x, [0, 2, 1, 3])
        return o

    def call(self, k, q, v, mask=None, training=False, **kwargs):
        batch_size, q_len, k_len = tf.shape(k)[0], tf.shape(q)[1], tf.shape(k)[1]
        q = self.drop(q, training=training)
        dq = self.dense_q(q)
        dk = self.dense_k(k)
        dv = self.dense_v(v)
        # print('dq1',dq.shape,'dk',dk.shape,'dv',dv.shape)
        # dq shape = (sequence_length,d_model)
        dq = self.split_heads(dq, batch_size, q_len, self.qk_depth)
        dk = self.split_heads(dk, batch_size, k_len, self.qk_depth)
        dv = self.split_heads(dv, batch_size, k_len, self.v_depth)
        # print('dq2', dq.shape, 'dk', dk.shape, 'dv', dv.shape)
        scaled_attention, attention_weights = scaled_dot_product_attention(dk, dq, dv, mask)
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, [batch_size, q_len, self.v_units])
        output = self.dense_o(concat_attention)
        return output


backbone_model = build_backbone_model()


def train():
    x_inputs_2cf, y_labels_2cf, _ = get_2class_emotion_dataset('datasets/emotion_2classify/weibo_senti_100k.csv',
                                                               MIN_SEQ_LEN, MAX_SEQ_LEN)
    seq_train_2cf, seq_test_2cf, label_train_2cf, label_test_2cf = train_test_split(x_inputs_2cf, y_labels_2cf,
                                                                                    train_size=0.8, random_state=14)
    '''two_classification_model = Sequential([InputLayer((MAX_SEQ_LEN,), dtype=tf.int32),
                                           backbone_model,
                                           Dense(2, name='2cf_out')])
    two_classification_model.compile(optimizer=Adam(learning_rate=CosineDecayWarmup(1e-4, 10000, 2000, 1e-4)),
                                     loss=SparseCategoricalCrossentropy(from_logits=True), metrics='accuracy')
    two_classification_model.fit(seq_train_2cf, label_train_2cf, validation_data=(seq_test_2cf, label_test_2cf),
                                 epochs=4)'''

    seq_train_6cf, label_train_6cf, _ = get_6class_emotion_dataset('datasets/emotion_5classify/usual_train.csv',
                                                                   MIN_SEQ_LEN, MAX_SEQ_LEN)
    seq_test_6cf, label_test_6cf, _ = get_6class_emotion_dataset('datasets/emotion_5classify/usual_eval_labeled.csv',
                                                                 MIN_SEQ_LEN, MAX_SEQ_LEN)
    five_classification_model = Sequential([InputLayer((MAX_SEQ_LEN,), dtype=tf.int32),
                                            backbone_model,
                                            Dense(6, name='6cf_out')])

    five_classification_model.compile(optimizer=Adam(learning_rate=CosineDecayWarmup(1e-4, 10000, 2000, 1e-4)),
                                      loss=SparseCategoricalCrossentropy(from_logits=True), metrics='accuracy')
    five_classification_model.fit(seq_train_6cf, label_train_6cf, validation_data=(seq_test_6cf, label_test_6cf),
                                  epochs=8)

    save_model(backbone_model, './models/emotion_backbone', include_optimizer=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', default='datasets')
    parser.add_argument('--attention-units', default=128)
    parser.add_argument('--attention-heads', default=4)
    parser.add_argument('--rnn-units', default=384)
    parser.add_argument('--dropout-rate', default=0.2)
    parser.add_argument('--vocab-size', default=21128)
    argsV = parser.parse_args()
    train()
