import argparse
import os.path
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Model, Input
from keras.layers import Layer, Dense, LayerNormalization, Dropout,concatenate
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.saving.save import load_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from bilstm_model import CosineDecayWarmup
from data_loader import get_rumor_type_a_classification_dataset,get_rumor_type_b_classification_dataset,xlnet_tokenizer,rumor_label_examples,bilstm_tokenizer
from shared_blocks import SelfRelMultiHeadAttention, RelativeSinusoidalPositionalEncoding, GlobalAttention
from tensorflow_addons.optimizers import MultiOptimizer

#tf.debugging.enable_check_numerics()

# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")

    return np.asfarray(data, float)

# 不同长度数据，统一为一个标准，倍乘x轴
def multiple_equal(x, y):
    x_len = len(x)
    y_len = len(y)
    times = x_len/y_len
    y_times = [i * times for i in y]
    return y_times
"""
    实现自定义Embedding层 自定义的embedding层用于加载xlnet或者ernie预训练权重
"""
class CustomEmbedding(Layer):
    def __init__(self, vocab_size, hidden_size, *args, **kwargs):
        super(CustomEmbedding, self).__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.embedding = self.add_weight(shape=([vocab_size, hidden_size]))

    def call(self, inputs, training=False, *args, **kwargs):
        hidden = tf.nn.embedding_lookup(self.embedding, inputs)
        return hidden

    def assign_embedding_weight(self, weight_np):
        self.embedding.assign(weight_np)

    def build(self, input_shape):
        return input_shape + (self.hidden_size,)

"""
 Transformer的前馈层
"""
class FeedforwardNetwork(Layer):
    def __init__(self, hidden_size, expand_hidden, dropout, *args, **kwargs):
        super(FeedforwardNetwork, self).__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.expand_hidden = expand_hidden
        self.dense1 = Dense(self.expand_hidden, use_bias=False, activation='relu')
        self.dense2 = Dense(self.hidden_size, use_bias=False)
        self.drop1 = Dropout(dropout)

    def call(self, inputs, training=False, *args, **kwargs):
        x = self.dense1(inputs)
        x = self.drop1(x, training=training)
        x = self.dense2(x)
        return x


"""
    transformerEncoder层 使用自主力机制
"""
class TransformerEncoderLayer(Layer):
    def __init__(self, d_model, n_heads, rel_pos, ffn_dim, dropout, *args, **kwargs):
        super(TransformerEncoderLayer, self).__init__(*args, **kwargs)
        self.rmha = SelfRelMultiHeadAttention(d_model, n_heads, False, rel_pos) #自注意力层
        self.ln1 = LayerNormalization(epsilon=1e-7, name='ln1')
        self.drop1 = Dropout(dropout, name='drop1')
        self.ln2 = LayerNormalization(epsilon=1e-7, name='ln2')
        self.ffn = FeedforwardNetwork(d_model, ffn_dim, dropout, name='ffn') #前馈层

    def call(self, inputs, att_masks=None, training=False, *args, **kwargs):
        x_rmha, att_weights = self.rmha(inputs, att_mask=att_masks, training=training)
        x_rmha = self.drop1(x_rmha,training=training)
        x = self.ln1(inputs + x_rmha)
        x_ffn = self.ffn(x)
        x = self.ln2(x + x_ffn)
        return x

    def build(self, input_shape):
        return input_shape


class RumorClassificationModel(Model):
    def __init__(self, vocab_size, emb_size, d_model, ffn_hidden, max_length, dropout=0.2, n_layers=4, n_heads=8,
                 n_classes=2):
        super(RumorClassificationModel, self).__init__()
        self.embedding = CustomEmbedding(vocab_size, emb_size)
        self.projection = Dense(d_model, use_bias=False)
        self.pos_enc = RelativeSinusoidalPositionalEncoding(max_length, d_model, name='rel_pos_layer')
        #self.out = Dense(n_classes, name='out_layer')
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, n_heads, self.pos_enc, ffn_hidden, dropout, name=f'encoder_{i}') for i in
            range(n_layers)]
        self.global_att = GlobalAttention(name='global_att')

    def assign_embedding(self, emb_array):
        self.embedding.assign_embedding_weight(emb_array)

    """
        embedding层 加载预训练的embedding权重
        projection层主要是解决 embedding层的数据维度和接下来的Transformer维度不匹配的问题.
    """
    def call(self, inputs, att_masks=None, training=None, **kwargs):
        x = self.embedding(inputs, training=training)
        x = self.projection(x)
        if att_masks is not None:
            if len(tf.shape(att_masks)) == 2:
                att_masks = tf.expand_dims(tf.expand_dims(att_masks, axis=1), axis=1)

            att_masks = tf.cast(att_masks, dtype=x.dtype)
        for layer in self.encoder_layers:
            #自注意力层
            x = layer(x, att_masks=att_masks, training=training)
        #q_mask = tf.expand_dims(1 - att_masks[:, 0, 0, :], axis=-1)
        #计算每个token的注意力权重分数 加权计算全局输出
        o = self.global_att(x,training=training,att_masks=att_masks)
        #o = tf.math.divide_no_nan(tf.reduce_sum(x * q_mask, axis=1), tf.reduce_sum(q_mask, axis=1))
        #o = self.out(x)
        return o


MIN_SEQ_LEN = 12
MAX_SEQ_LEN = 96



NUM_CLASSES = 2
def load_emotion_backbone(_args):
    if not os.path.exists(_args.emotion_model_path) or not os.path.isdir(_args.emotion_model_path):
        print(f'在{_args.emotion_model_path}文件夹下没有找到已保存情绪模型,请指定模型存放的文件夹')
    model = load_model(_args.emotion_model_path)
    return model

"""
    这里的作用是将 评论情感分类模型的输出和博文内容编码器的输出进行合并
    得到最终输出
"""
'''def build_merge_model(_args):
    content_input = Input(shape=(_args.d_model,),dtype=tf.float32)
    emotion_input = Input(shape=(_args.emotion_hidden,),dtype=tf.float32)
    x = concatenate([content_input,emotion_input],axis=-1)
    x = Dropout(_args.dropout_rate)(x)
    x = Dense(_args.merge_hidden,activation='relu')(x)
    x = LayerNormalization(epsilon=1e-7)(x)
    o = Dense(NUM_CLASSES)(x)
    return Model(inputs=[content_input,emotion_input],outputs=o)'''

def train(_args):
    #训练函数
    @tf.function
    def train_step(content_seq, content_mask, x_label):
        with tf.GradientTape() as tape:
            encoder_logits = rumor_encoder_model(content_seq, content_mask, training=True)
            #emotion_logits = tf.stop_gradient(emotion_backbone_model(comment_seq,training=False))
            #emotion_logits = emotion_backbone_model(comment_seq,training=False)
            #outputs = merge_model([encoder_logits,emotion_logits])
            loss = loss_obj(x_label, encoder_logits)

        grads = tape.gradient(loss,trainable_weights)

        opt.apply_gradients(zip(grads, trainable_weights))
        acc_item = tf.equal (tf.argmax(encoder_logits, axis=-1, output_type=x_label.dtype), x_label)
        acc = tf.reduce_sum(tf.cast(acc_item, dtype=tf.float32)) / tf.cast(tf.shape(acc_item)[0], tf.float32)
        return loss, acc

    @tf.function
    def eval_step(content_seq, content_mask, x_label):
        encoder_logits = rumor_encoder_model(content_seq, content_mask, training=True)
        #emotion_logits = tf.stop_gradient(emotion_backbone_model(comment_seq,training=False))
        #outputs = merge_model([encoder_logits,emotion_logits])
        loss = loss_obj(x_label, encoder_logits)
        acc_item = tf.equal(tf.argmax(encoder_logits, axis=-1, output_type=x_label.dtype), x_label)
        acc = tf.reduce_sum(tf.cast(acc_item, dtype=tf.float32)) / tf.cast(tf.shape(acc_item)[0], tf.float32)
        pred_labels = tf.math.argmax(encoder_logits, axis=-1)  # 获取预测标签
        return loss, acc, pred_labels  # 返回预测标签
    vocab_size = _args.xlnet_vocab_size if _args.used_embedding == 'xlnet' else _args.ernie_vocab_size
    """
        这个模型的作用是 理解博文内容 并且输出一个向量
    """
    rumor_encoder_model = RumorClassificationModel(vocab_size=vocab_size,
                                                   emb_size=_args.embedding_size,
                                                   d_model=_args.d_model, ffn_hidden=_args.ffn_hidden,
                                                   max_length=_args.sequence_length)
    embedding_filepath = _args.xlnet_embedding_filepath if _args.used_embedding == 'xlnet' else _args.ernie_embedding_filepath
    if os.path.exists(embedding_filepath):

        rumor_encoder_model.assign_embedding(np.load(embedding_filepath))
        print('成功从文件中加载了预训练的embedding权重')
    else:
        print('没有识别到预训练权重,使用随机权重进行初始化')
    emotion_backbone_model = load_emotion_backbone(_args)
    #merge_model = build_merge_model(_args)

    trainable_weights = rumor_encoder_model.trainable_weights
    loss_obj = SparseCategoricalCrossentropy(from_logits=True)
    """
        是否启用分层学习率 如果启用分层学习率，embedding的学习率将是 其它层的 10%
    """
    if _args.enable_discriminative_lr:
        emb_lr = CosineDecayWarmup(_args.learning_rate * 0.1, _args.decay_steps, 1000, 1e-4)
        others_lr = CosineDecayWarmup(_args.learning_rate, _args.decay_steps, 1000, 1e-4)

        opt_emb = Adam(learning_rate=emb_lr)
        opt_other = Adam(learning_rate=others_lr)

        opt = MultiOptimizer([(opt_emb,rumor_encoder_model.layers[0]),(opt_other,rumor_encoder_model.layers[1:] )])
    else:
        lr = CosineDecayWarmup(_args.learning_rate, _args.decay_steps, 1000, 1e-4)
        opt = Adam(learning_rate=lr)
    get_rumor_classification_dataset = get_rumor_type_a_classification_dataset \
        if _args.dataset_type =='typeA' else get_rumor_type_b_classification_dataset

    writer = tf.summary.create_file_writer('./logs')
    eval_loss = []
    eval_acc = []
    for n in range(0,_args.train_epochs):
        print(f'开始第{n+1}轮训练:')
        content_train, label_train, content_mask_train, comment_train, comment_mask_train = get_rumor_classification_dataset(
            _args.train_filepath, _args.min_sequence_length, _args.sequence_length, _args.n_comment_keeps)
        content_test, label_test, content_mask_test, comment_test, comment_mask_test = get_rumor_classification_dataset(
            _args.test_filepath, _args.min_sequence_length, _args.sequence_length, _args.n_comment_keeps)
        n_train_total_steps = len(content_train) // _args.batch_size
        n_eval_total_steps = len(content_test) // _args.batch_size
        with tqdm(total=n_train_total_steps) as bar:
            for i in range(n_train_total_steps):
                """
                    从数据集上剪切BATCH_SIZE大小的数据集 用于训练
                """
                batch_content_train = content_train[i*_args.batch_size:(i+1)*_args.batch_size]
                batch_label_train = label_train[i*_args.batch_size:(i+1)*_args.batch_size]
                batch_content_mask_train = content_mask_train[i*_args.batch_size:(i+1)*_args.batch_size]
                '''batch_comment_train = comment_train[i*_args.batch_size:(i+1)*_args.batch_size]
                batch_comment_mask_train = comment_mask_train[i*_args.batch_size:(i+1)*_args.batch_size]'''

                #训练部分
                train_batch_loss, train_bach_acc = train_step(batch_content_train, batch_content_mask_train, batch_label_train)
                with writer.as_default(step=n*n_train_total_steps + i):
                    tf.summary.scalar('train_loss', train_batch_loss)
                    tf.summary.scalar('acc', train_bach_acc)
                if i % 100 == 0:
                    writer.flush()
                bar.set_postfix({
                                'epoch':n+1,
                                'train_loss':round(float(train_batch_loss), 4),
                                 'train_acc':round(float(train_bach_acc), 4)})

                bar.update()
        '''eval_acc_sum, eval_loss_sum = 0., 0.
        with tqdm(total=n_eval_total_steps) as bar:

            for i in range(n_eval_total_steps):
                batch_content_test = content_test[i * _args.batch_size: (i + 1) * _args.batch_size]
                batch_label_test = label_test[i * _args.batch_size: (i + 1) * _args.batch_size]
                batch_content_mask_test = content_mask_test[i * _args.batch_size: (i + 1) * _args.batch_size]
                batch_comment_test = comment_test[i * _args.batch_size: (i + 1) * _args.batch_size]
                batch_comment_mask_test = comment_mask_test[i * _args.batch_size: (i + 1) * _args.batch_size]
                eval_batch_loss, eval_batch_acc = eval_step(
                    batch_content_test, batch_content_mask_test,
                    batch_label_test, batch_comment_test, batch_comment_mask_test)
                eval_loss_sum += float(eval_batch_loss)
                eval_acc_sum += float(eval_batch_acc)
                bar.set_postfix({'epoch': n + 1,
                                 'eval_loss': round(float(eval_batch_loss), 4),
                                 'eval_acc': round(float(eval_batch_acc), 4)})
                bar.update()
        eval_loss_avg = round(eval_loss_sum / n_eval_total_steps, 4)
        eval_acc_avg = round(eval_acc_sum / n_eval_total_steps, 4)
        print(f'epoch: {n + 1}/{_args.train_epochs},avg_eval_loss:{eval_loss_avg},avg_eval_acc: {eval_acc_avg}')'''

        #评估部分
        eval_acc_sum,eval_loss_sum =0.,0.
        eval_precision_sum, eval_recall_sum, eval_f1_sum = 0., 0., 0.

        # 定义一个函数，用于根据真实标签和预测标签计算查准率，查全率，调和平均值
        def calculate_metrics(true_labels, pred_labels):
            # 计算真阳性（TP），假阳性（FP），假阴性（FN）
            true_labels = tf.cast(true_labels, tf.int32)
            pred_labels = tf.cast(pred_labels, tf.int32)
            TP = sum([1 for t, p in zip(true_labels, pred_labels) if tf.reduce_all(t - p == 0)and tf.reduce_all(t == 1)])
            FP = sum([1 for t, p in zip(true_labels, pred_labels) if tf.reduce_all(t - p == -1)])
            FN = sum([1 for t, p in zip(true_labels, pred_labels) if tf.reduce_all(t - p == 1)])
            # 计算查准率，查全率，调和平均值
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            # 返回查准率，查全率，调和平均值
            return precision, recall, f1
        with tqdm(total=n_eval_total_steps) as bar:

            for i in range(n_eval_total_steps):
                batch_content_test = content_test[i * _args.batch_size: (i + 1) * _args.batch_size]
                batch_label_test = label_test[i * _args.batch_size: (i + 1) * _args.batch_size]
                batch_content_mask_test = content_mask_test[i * _args.batch_size: (i + 1) * _args.batch_size]
                '''batch_comment_test = comment_test[i * _args.batch_size: (i + 1) * _args.batch_size]
                batch_comment_mask_test = comment_mask_test[i * _args.batch_size: (i + 1) * _args.batch_size]'''
                eval_batch_loss, eval_batch_acc,pred_labels = eval_step(
                    batch_content_test, batch_content_mask_test,
                    batch_label_test)
                eval_loss_sum += float(eval_batch_loss)
                eval_acc_sum += float(eval_batch_acc)

                batch_label_test_re = tf.reshape(batch_label_test, [-1])
                # 在eval_step函数中，获取模型的输出
                pred_labels_re = tf.reshape(pred_labels, [-1])
                # 调用calculate_metrics函数时，使用pred_labels_re作为参数
                eval_batch_precision, eval_batch_recall, eval_batch_f1 = calculate_metrics(batch_label_test_re,
                                                                                pred_labels_re)

                eval_precision_sum += float(eval_batch_precision)
                eval_recall_sum += float(eval_batch_recall)
                eval_f1_sum += float(eval_batch_f1)
                bar.set_postfix({'epoch':n+1,
                                 'eval_loss':round(float(eval_batch_loss), 4),
                                 'eval_acc':round(float(eval_batch_acc), 4),
                                 'eval_precision': round(float(eval_batch_precision), 4),
                                 'eval_recall': round(float(eval_batch_recall), 4),
                                 'eval_f1': round(float(eval_batch_f1), 4)})
                bar.update()
            eval_loss_avg = round(eval_loss_sum / n_eval_total_steps,4)
            eval_acc_avg = round(eval_acc_sum / n_eval_total_steps,4)

            eval_loss.append(eval_loss_avg)
            eval_acc.append(eval_acc_avg)
                # 新增以下三行代码，用于计算并显示平均查准率，平均查全率，平均调和平均值
            eval_precision_avg = round(eval_precision_sum / n_eval_total_steps, 4)
            eval_recall_avg = round(eval_recall_sum / n_eval_total_steps, 4)
            eval_f1_avg = round(eval_f1_sum / n_eval_total_steps, 4)


            print(f'epoch: {n+1}/{_args.train_epochs},avg_eval_loss:{eval_loss_avg},avg_eval_acc: {eval_acc_avg},avg_eval_precision: {eval_precision_avg},avg_eval_recall: {eval_recall_avg},avg_eval_f1: {eval_f1_avg}')

    print('模型训练完成')
    plt.figure(figsize=(10, 5))
    plt.title("Testing Accuracy and Loss")
    plt.plot(eval_acc, label="acc")
    plt.plot(eval_loss, label="loss")
    plt.xlabel("epochs")
    plt.ylabel("accuracy and loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-length', default=96, type=int,help='用于预测的文本序列长度,超长会进行截断')

    parser.add_argument('--train-filepath', default='./datasets/rumor_A/Weibo_train.txt', type=str,help='设置训练文件路径')
    parser.add_argument('--dataset-type', default='typeA', type=str,choices=['typeA','typeB'],help='设置谣言数据集种类,可以选择typeA和typeB')

    parser.add_argument('--test-filepath', default='./datasets/rumor_A/Weibo_valid.txt', type=str,help='设置测试文件路径')
    parser.add_argument('--embedding-size', default=768, type=int,help='设置embedding层的输出尺度')
    parser.add_argument('--xlnet-embedding-filepath', default='./xlnet_embedding.npy', type=str,help='设置xlnet的embedding文件位置')
    parser.add_argument('--ernie-embedding-filepath', default='./ernie_embedding.npy', type=str,help='设置ernie的embedding文件位置')
    parser.add_argument('--d-model', default=128, type=int,help='设置模型自注意力的隐藏层宽度')
    parser.add_argument('--n-layers', default=6, type=int,help='设置模型层数')
    parser.add_argument('--batch-size', default=8, type=int,help='设置批次大小')
    parser.add_argument('--n-heads', default=4, type=int,help='设置多头注意力的头的个数')
    parser.add_argument('--train-epochs', default=10, type=int,help='设置训练的次数')
    parser.add_argument('--n-comment-keeps', default=4, type=int,help='设置每个博文保留多少条评论')
    parser.add_argument('--min-sequence-length', default=12, type=int,help='设置最短文本长度,如果低于这个长度的博文不会进行学习')
    parser.add_argument('--ffn-hidden', default=768, type=int,help='设置ffn层的隐藏层宽度')
    parser.add_argument('--merge-hidden', default=384, type=int,help='设置博文特征和评论情绪特征合并时候的隐藏层宽度')
    parser.add_argument('--xlnet-vocab-size', default=32000, type=int, help='设置xlnet词汇表大小')
    parser.add_argument('--ernie-vocab-size', default=40000, type=int, help='设置ernie词汇表大小')
    parser.add_argument('--learning-rate', default=0.0001, type=float,help='设置模型的学习率')
    parser.add_argument('--decay-steps', default=10000, type=int,help='设置学习的步数,用于模型学习率衰减')
    parser.add_argument('--dropout-rate', default=0.2, type=float,help='设置训练时隐藏层丢弃的比例，用于减少过拟合')
    parser.add_argument('--model-save-path', default='models/rumor', type=str,help='设置模型保存路径用于读取')
    parser.add_argument('--emotion-model-path', default='models/emotion_backbone', type=str,help='设置情感模型的保存路径,会进行读取')
    parser.add_argument('--emotion-hidden', default=384, type=int,help='设置情感模型输出的隐藏层宽度')
    parser.add_argument('--used-embedding', default='ernie', type=str,help='设置要使用的embedding来源 可以选择xlnet 或者 ernie')
    parser.add_argument("--enable-discriminative-lr",default=False,type=bool,help="是否启用分层学习率，启用后会为embedding设置更小的学习率")
    argsV = parser.parse_args()
    train(argsV)
