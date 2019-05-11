import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet101
import torchvision.transforms as T
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from generator import Attention



class Discriminator(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, attention_dim, encoder_dim=2048, dropout=0.2, discriminator_path = 'data/discriminator_params.pkl'):
        super(Discriminator, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.rnn_cell = nn.LSTMCell(encoder_dim + embedding_size, hidden_size, bias=True)
        self.classifier = nn.Linear(hidden_size, 1)
        self.gru2hidden = nn.Linear(2 * 2 * hidden_size, hidden_size)

        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)

        self.hidden2out = nn.Linear(hidden_size, 1)

        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.h_fc = nn.Linear(encoder_dim, hidden_size)
        self.c_fc = nn.Linear(encoder_dim, hidden_size)
        self.f_beta = nn.Linear(hidden_size, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.log_every = 10
        self.save_every = 100
        self.discriminator_path = discriminator_path
        if os.path.exists(self.discriminator_path):
            print('Start loading discriminator')
            self.load_state_dict(torch.load(self.discriminator_path))



    def forward(self, inputs, hidden):
        '''
            inputs: (batch_size, max_length)
        '''
        inputs = self.embeddings(inputs) # (batch_size, max_length, embedding_size)
        _, hidden = self.rnn(inputs) # hidden: (batch_size, 4, hidden_size)
        hidden = hidden.contiguous() # (batch_size, 4, hidden_size)
        outputs = self.gru2hidden(hidden.view(-1, 4 * self.hidden_size)) # (batch_size, hidden_size)
        
        outputs = torch.tanh(outputs)
        outputs = self.dropout(outputs)
        outputs = self.hidden2out(outputs) # (batch_size, 1)
        outputs = torch.sigmoid(outputs) # (batch_size, 1)

        return outputs.squeeze(1) # (batch_size,)


    def predict(self, features, captions, lengths, device):
        decoder_lengths = [length - 1 for length in lengths]

        batch_size = features.size(0)

        y_predicted = torch.zeros(batch_size, max(decoder_lengths)).to(device)

        mean_features = features.mean(dim=1) # (batch_size, encoder_dim)
        hidden_state = self.h_fc(mean_features) # (batch_size, lstm_size)
        cell_state = self.c_fc(mean_features) # (batch_size, lstm_size)

        embeddings = self.embeddings(captions)

        for step in range(max(decoder_lengths)):
            curr_batch_size = sum([l > step for l in decoder_lengths])

            attention_weighted_encoding, _ = self.attention(features[:curr_batch_size], hidden_state[:curr_batch_size]) # (curr_batch_size, encoder_dim)

            gate = self.sigmoid(self.f_beta(hidden_state[:curr_batch_size])) # (curr_batch_size, encoder_dim)

            attention_weighted_encoding = gate * attention_weighted_encoding # (curr_batch_size, encoder_dim)
            hidden_state, cell_state = self.rnn_cell(torch.cat([embeddings[:curr_batch_size, step, :], attention_weighted_encoding], dim=1), (hidden_state[:curr_batch_size], cell_state[:curr_batch_size]))
            y_pred = self.classifier(self.dropout(hidden_state)) # (batch_size, 1)
            y_pred = torch.sigmoid(y_pred)
            y_predicted[:curr_batch_size, step] = y_pred.squeeze(1)
        
        #y_predicted = y_predicted.sum(dim=1) / lengths # (batch_size,)
        y_predicted = y_predicted.sum(dim=1) / torch.FloatTensor(decoder_lengths).to(device)

        return y_predicted


    def pre_train(self, generator, dataloader, num_epochs, vocab, num_batches=None, alpha_c=1.0):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_steps = len(dataloader)

        for epoch in range(num_epochs):
            for step, (imgs, captions, lengths) in enumerate(dataloader):
                imgs = imgs.to(device)
                captions = captions.to(device) # (batch_size, batch_max_length)

                batch_size = captions.size(0)

                features = generator.encoder(imgs)
                features = features.view(features.size(0), -1, features.size(-1))
                captions_pred = generator.sample(features, vocab) # list, (batch_size, var_length). eg: [[1, 4, ... , 19, 2]], containing <sos> and <eos>
                # sort captions_pred, features
                sorted_indices, captions_pred = zip(*sorted(enumerate(captions_pred), key=lambda x: len(x[1]), reverse=True))
                sorted_indices = list(sorted_indices)
                captions_pred = list(captions_pred)
                sorted_features = features[list(sorted_indices)]


                lengths_pred = [len(caption_pred) for caption_pred in captions_pred]
                max_length_pred = lengths_pred[0]
                for index, caption_pred in enumerate(captions_pred):
                    captions_pred[index] = caption_pred + [0] * (max_length_pred - len(caption_pred))
                
                captions_pred = torch.LongTensor(captions_pred).to(device)
                #lengths_pred = list()
                #for i in range(batch_size):
                    #lengths_pred = len(captions_pred[i]) if len(tmp_captions_pred[i]) < captions.size(1) else captions.size(1)

                #captions, _ = pack_padded_sequence(captions, lengths, batch_first=True)

                #captions_pred, _ = pack_padded_sequence(captions_pred, max_lengths_pred, batch_first=True)
                
                batch_size = features.size(0)
                num_pixels = features.size(1)
                
                #-------------------------- RUN RNN ---------------------------------------
                D_real = self.predict(features, captions, lengths, device) # (batch_size, )
                D_fake = self.predict(sorted_features, captions_pred, lengths_pred, device) # (batch_size, )
                loss = - torch.mean(torch.log(D_real)) - torch.mean(torch.log(1 - D_fake))
    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % self.log_every == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, num_epochs, step, num_steps, loss.item(), np.exp(loss.item())))
                    print('mean(D_real):', torch.mean(D_real).item(), 'mean(D_fake):', torch.mean(D_fake).item())
                if (step + 1) % self.save_every == 0:
                    print('Start saving discriminator')
                    torch.save(self.state_dict(), self.discriminator_path)
                if num_batches and step + 1 >= num_batches:
                    break
