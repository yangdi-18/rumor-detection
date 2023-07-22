import csv
import json
import os.path
import random
import re

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

bilstm_tokenizer = Tokenizer.from_pretrained("bert-base-chinese")
xlnet_tokenizer = Tokenizer.from_pretrained("hfl/chinese-xlnet-base")

six_emotion_label_examples = ['neutral', 'sad', 'angry', 'surprise', 'fear', 'happy']
rumor_label_examples =['real','fake']
def re_map(_text):
    _text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', _text)
    _text = re.sub('\\[[\\u4e00-\\u9fa5]{1,6}]', '', _text)
    _text = re.sub('(@[\\u4e00-\\u9fa5a-zA-Z0-9_-]{4,30})', '', _text)
    # _text_sub = re.sub('(@[\\u4e00-\\u9fa5a-zA-Z0-9_-]{4,30}?)(:?)($|\\s|@|\\(|\\[|（|【)','\g<3>',_text)
    # while _text_sub != _text:
    # _text = _text_sub
    # _text_sub = re.sub('(@[\\u4e00-\\u9fa5a-zA-Z0-9_-]{4,30}?)(:?)($|\\s|@|\\(|\\[|（|【)', '\g<3>', _text)
    _text = re.sub(r'(.+?)\1+', '\g<1>', _text)
    _text = re.sub(r'\(\s*\)|\[\s*]|（\s*）|【\s*】]', '', _text)
    _text = _text.replace('回复自', '').replace('回复', '').replace('/', '')
    _text = re.sub(r'([^\u4e00-\u9fa5a-zA-Z0-9]|^)(\s+)([^\u4e00-\u9fa5a-zA-Z0-9]|$)|'
                   r'([\u4e00-\u9fa5a-zA-Z0-9]|^)(\s+)([^\u4e00-\u9fa5a-zA-Z0-9]|$)|'
                   r'([^\u4e00-\u9fa5a-zA-Z0-9]|^)(\s+)([\u4e00-\u9fa5a-zA-Z0-9]|$)', '\g<1>\g<3>', _text)
    return _text

"""
    加载2分类情感数据集,用来学习评论的情感特征.
    返回 评论内容,下表索引,评论内容掩码
"""
def get_2class_emotion_dataset(csv_path, min_seq_length, pad_seq_length):
    tokenizer = bilstm_tokenizer
    label_list = []
    seq_list = []
    mask_list = []
    tokenizer.enable_truncation(max_length=pad_seq_length)
    tokenizer.enable_padding(length=pad_seq_length)
    with open(csv_path, encoding='utf-8') as fi:
        items = fi.read().split('\n')
        if items[-1] == '': items.pop(-1)
        items.pop(0)
    for item in tqdm(items):
        label = item[0]
        seq = re_map(item[2:])
        if len(seq) < min_seq_length:
            continue
        label_list.append(int(label))
        token = tokenizer.encode(seq)
        seq_list.append(token.ids)
        mask_list.append(token.attention_mask)
    label_array = np.array(label_list, dtype=np.int32)
    seq_array = np.array(seq_list, dtype=np.int32)
    mask_array = 1 - np.array(mask_list, dtype=np.int32)
    return seq_array, label_array, mask_array
"""
    加载6分类情感数据集,用来学习评论的情感特征.
    返回 评论内容,下表索引,评论内容掩码
"""

def get_6class_emotion_dataset(csv_path, min_seq_length, pad_seq_length):
    tokenizer = bilstm_tokenizer
    label_list = []
    seq_list = []
    mask_list = []
    tokenizer.enable_truncation(max_length=pad_seq_length)
    tokenizer.enable_padding(length=pad_seq_length)
    with open(csv_path, encoding='utf-8') as fi:
        for row in csv.DictReader(fi, skipinitialspace=True):
            if len(row['文本']) < min_seq_length:
                continue
            label_idx = six_emotion_label_examples.index(row['情绪标签'])
            label_list.append(label_idx)
            token = tokenizer.encode(row['文本'])
            seq_list.append(token.ids)
            mask_list.append(token.attention_mask)
    label_array = np.array(label_list, dtype=np.int32)
    seq_array = np.array(seq_list, dtype=np.int32)
    mask_array = 1 - np.array(mask_list, dtype=np.int32)
    return seq_array, label_array, mask_array
"""
    加载datasets/rumor_B文件夹下的数据集.
    返回依次是 博文内容,标签下标 0代表非谣言 1代表谣言,博文内容掩码,评论内容,评论内容掩码
"""
def get_rumor_type_b_classification_dataset(json_path,min_seq_length,pad_seq_length,n_comment_keep):
    tokenizer = xlnet_tokenizer
    with open(json_path,mode='r',encoding='utf-8') as fi:
        entities = json.load(fi)
    x_content_list = []
    x_content_mask_list = []
    x_comment_list = []
    x_comment_mask_list = []
    y_label_list = []
    tokenizer.enable_truncation(max_length=pad_seq_length)
    tokenizer.enable_padding(length=pad_seq_length)

    for entity in entities:
        if 'content_aug' in entity:
            content = re_map(random.choice(entity['content_aug']))
        else:
            content = re_map(entity['content'])
        comments = [re_map(comment) for comment in entity['comments']]
        comments = sorted(comments,reverse=True,key=lambda  k :len(k))
        comment = ' '.join(comments[:n_comment_keep])
        if len(content) < min_seq_length or len(comment) < min_seq_length:
            continue
        content_token = tokenizer.encode(content)
        comment_token = tokenizer.encode(comment)
        x_content_list.append(content_token.ids)
        x_comment_list.append(comment_token.ids)
        x_content_mask_list.append(comment_token.attention_mask)
        x_comment_mask_list.append(comment_token.attention_mask)
        y_label_list.append(rumor_label_examples.index(entity['label']))
    idx = np.array([i for i in range(len(y_label_list))])
    np.random.shuffle(idx)
    content_np = np.array(x_content_list,dtype=np.int32)
    comment_np = np.array(x_comment_list,dtype=np.int32)
    content_mask_np =1 - np.array(x_content_mask_list,dtype=np.int32)
    comment_mask_np =1 - np.array(x_comment_mask_list,dtype=np.int32)
    label_np =  np.array(y_label_list,dtype=np.int32)
    return content_np[idx,:],label_np[idx],content_mask_np[idx,:],comment_np[idx,:],comment_mask_np[idx,:]


"""
    加载datasets/rumor_A文件夹下的数据集.
    返回依次是 博文内容,标签下标 0代表非谣言 1代表谣言,博文内容掩码,评论内容,评论内容掩码

"""
def get_rumor_type_a_classification_dataset(label_path,min_seq_length,pad_seq_length,n_comment_keep):
    tokenizer = xlnet_tokenizer
    base_dir = os.path.dirname(label_path)
    with open(label_path,mode='r',encoding='utf-8') as fi:
        lines = fi.read().split('\n')

    tokenizer.enable_truncation(max_length=pad_seq_length)
    tokenizer.enable_padding(length=pad_seq_length)
    x_content_list = []
    x_content_mask_list = []
    x_comment_list = []
    x_comment_mask_list = []
    y_label_list = []

    for line in tqdm(lines):
        splits = line.split('\t')
        if len(splits) != 3:
            continue
        event_id = splits[0][4:]
        label_idx = int(splits[1][6:])
        event_path = os.path.join(base_dir,'Weibo',event_id+'.json')
        with open(event_path,encoding='utf-8') as fi:
            event_dict = json.load(fi)
        seq_content = event_dict[0]['text']
        seq_comments = [comment['text'] if comment['text'] != '转发微博' else '' for comment in event_dict[1:]]
        seq_comments = sorted(seq_comments,reverse=True,key=lambda t:len(t))
        if len(seq_comments) == 0:
            print(event_id)
        seq_comment = random.choice(seq_comments[:n_comment_keep])
        if len(seq_content) < min_seq_length or len(seq_comment) < min_seq_length:
            continue
        content_token = tokenizer.encode(seq_content)
        comment_token = tokenizer.encode(seq_comment)
        x_content_list.append(content_token.ids)
        x_comment_list.append(comment_token.ids)
        x_content_mask_list.append(comment_token.attention_mask)
        x_comment_mask_list.append(comment_token.attention_mask)
        y_label_list.append(label_idx)
    idx = np.array([i for i in range(len(y_label_list))])
    np.random.shuffle(idx)
    content_np = np.array(x_content_list, dtype=np.int32)
    comment_np = np.array(x_comment_list, dtype=np.int32)
    content_mask_np = 1 - np.array(x_content_mask_list, dtype=np.int32)
    comment_mask_np = 1 - np.array(x_comment_mask_list, dtype=np.int32)
    label_np = np.array(y_label_list, dtype=np.int32)
    return content_np[idx, :], label_np[idx], content_mask_np[idx, :], comment_np[idx, :], comment_mask_np[idx, :]

