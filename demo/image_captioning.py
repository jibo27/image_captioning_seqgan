import torch
import torch.utils.data as data
import os
import pickle
import numpy as np

from generator import Generator
import torch 

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)




def image_captioning(img_path):
    embedding_size = 512
    lstm_size = 512
    attention_dim = 512

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
       
    vocab_size = len(vocab)
    
    generator = Generator(attention_dim, embedding_size, lstm_size, vocab_size, load_path='./generator_params.pkl', noise=True, device='cpu')
    generator = generator.to(device)
    generator = generator.eval()
    
    caption_set = set()
    max_iter = 50
    while len(caption_set) != 3 and max_iter != 0: # must generate 3 unique results
        caption_set.add(generator.inference(vocab, img_path=img_path, translate_flag=True))
        max_iter -= 1
    return caption_set


    
