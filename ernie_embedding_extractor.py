import argparse

import numpy as np
from transformers import ErnieForMaskedLM


from tokenizers import Tokenizer
"""
    从ERNIE3.0模型上提取权重.
"""
def extract_embedding_from_ernie(ernie_path,ernie_name_scope):
    model = ErnieForMaskedLM.from_pretrained(ernie_path)
    embedding = model.base_model.embeddings.word_embeddings.weight.detach().numpy()
    return embedding
    #print(embedding)
    #print()
    #print(embedding)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ernie-path', default='nghuyong/ernie-3.0-base-zh', type=str,help='')
    parser.add_argument('--ernie-embedding-namescope', default='./xlnet_ckpt/xlnet_model.ckpt', type=str)
    parser.add_argument("--ernie-output-path",default="./ernie_embedding.npy",type=str)
    #parser.add_argument("--ernie-output-path",default="")


    args = parser.parse_args()
    embedd = extract_embedding_from_ernie(args.ernie_path,args.ernie_embedding_namescope)
    np.save(args.ernie_output_path,embedd)

    #embedding_np = get_embedding_and_tokenizer(args.xlnet_embedding_path)
    #np.save(args.embedding_save_path,embedding_np)
    #print(embedding_np.shape)
    print(f'embedding提取成功,已经保存到{args.ernie_output_path}文件中')

