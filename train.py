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


def main(args):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab)
    print('vocab_size:', vocab_size)

    dataloader = get_loader(image_dir, caption_path, vocab, 
                            args.batch_size,
                            crop_size,
                            shuffle=True, num_workers=num_workers)

    generator = Generator(attention_dim, embedding_size, lstm_size, vocab_size, load_path=args.g_path, noise=args.noise)
    generator = generator.to(device)
    generator = generator.train()

    discriminator = Discriminator(vocab_size, embedding_size, lstm_size, attention_dim, load_path=args.d_path)
    discriminator = discriminator.to(device)
    discriminator = discriminator.train()


    

    
    if args.train_mode == 'gd':
        for _ in range(5):
            for i in range(4):
                generator.pre_train(dataloader, vocab)
            for i in range(1):
                discriminator.fit(generator, dataloader, vocab)
    elif args.train_mode == 'dg':
        discriminator.fit(generator, dataloader, vocab)
        generator.pre_train(dataloader, vocab)
    elif args.train_mode == 'd':
        discriminator.fit(generator, dataloader, vocab)
    elif args.train_mode == 'g':
        generator.pre_train(dataloader, vocab)
    elif args.train_mode == 'ad':
        for i in range(5):
            generator.ad_train(dataloader, discriminator, vocab, gamma=args.gamma, update_every=args.update_every, alpha_c=1.0, num_rollouts=args.num_rollouts)

#    for i in range(5):
#        print("D")
#        discriminator.fit(generator, dataloader, vocab, num_batches=100)
#        print("G")
#        generator.ad_train(dataloader, discriminator, vocab, num_batches=20, alpha_c=1.0)

    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', type=str, default='gd', help='mode of pre-train')
    parser.add_argument('--g_path', type=str, default=None, help='')
    parser.add_argument('--d_path', type=str, default=None, help='')
    parser.add_argument('--gamma', type=float, default=2.0, help='')
    parser.add_argument('--update_every', type=int, default=20, help='')
    parser.add_argument('--batch_size', type=int, default=16, help='') # Colab cannot run other codes if batch_size is set to be 16
    parser.add_argument('--noise', type=bool, default=False, help='')
    parser.add_argument('--num_rollouts', type=int, default=16, help='')
    args = parser.parse_args()
    main(args)

