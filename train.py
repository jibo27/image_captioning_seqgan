import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from dataloader import get_loader
from generator import Generator
from discriminator import Discriminator
from settings import *


def train():
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab)
    print('vocab_size:', vocab_size)

    dataloader = get_loader(image_dir, caption_path, vocab, 
                            batch_size,
                            crop_size,
                            shuffle=True, num_workers=num_workers)

    generator = Generator(attention_dim, embedding_size, lstm_size, vocab_size)
    generator = generator.to(device)
    generator = generator.train()

    discriminator = Discriminator(vocab_size, embedding_size, lstm_size, attention_dim)
    discriminator = discriminator.to(device)
    discriminator = discriminator.train()



    for i in range(5):
        discriminator.pre_train(generator, dataloader, 1, vocab)
        generator.pre_train(dataloader, 1, vocab)


#    for i in range(5):
#        print("D")
#        discriminator.pre_train(generator, dataloader, 1, vocab, num_batches=100)
#        print("G")
#        generator.ad_train(dataloader, discriminator, vocab, 1, num_batches=20, alpha_c=1.0)

    
train()

