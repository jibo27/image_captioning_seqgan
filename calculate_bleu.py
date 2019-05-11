import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from build_vocab import Vocabulary
from pycocotools.coco import COCO
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.transforms as T
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu
import tqdm

from dataloader import get_loader
from generator import Generator
from settings import *


with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)
print('vocab_size:', vocab_size)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

#image_dir = './data/valresized2014/'
#caption_path = './data/annotations/captions_val2014.json'
num_workers = 32


dataloader = get_loader(image_dir, caption_path, vocab, 
                        batch_size,
                        crop_size,
                        shuffle=False, num_workers=num_workers, transform=transform)


generator = Generator(attention_dim, embedding_size, lstm_size, vocab_size, load_ad=False)
generator = generator.to(device)
generator = generator.eval()

def translate(indices, vocab):
    sentences = list()
    for index in indices:
        word = vocab.idx2word[int(index)]
        if word == '<eos>':
            break
        sentences.append(word)
    return ' '.join(sentences)

scores = list()
num_batches = 100
print('total length:', len(dataloader), '; we chose %d batches'%(num_batches))
for index, (imgs, captions, lengths) in tqdm.tqdm(enumerate(dataloader)):
    imgs = imgs.to(device)

    features = generator.encoder(imgs)
    indices_list = generator.sample(features, vocab)
    for i in range(len(indices_list)):
        sentence_pred = translate(indices_list[i][1:], vocab)
        sentence = translate(captions[i][1:], vocab)
        bleus = list()
        for j in range(4):
            weights = [0] * 4
            weights[j] = 1
            bleus.append(sentence_bleu([sentence], sentence_pred, weights=weights))
        scores.append(bleus)
    if index + 1 == num_batches:
        break

scores = np.asarray(scores)
print(scores.shape)

for i in range(4):
    print("BLEU{}".format(i + 1))
    print('mean score:', np.sum(scores[:, i]) / scores.shape[0])
    print('min score:', np.min(scores[:, i]))
    print('max score:', np.max(scores[:, i]))
    print('sum score:', np.sum(scores[:, i]))
