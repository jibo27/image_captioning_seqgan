import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet101
import torchvision.transforms as T
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

class Encoder(torch.nn.Module):
    def __init__(self, enc_img_size=14):
        super(Encoder, self).__init__()
        resnet = resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((enc_img_size, enc_img_size))
        self.fine_tune()


    def forward(self, imgs):
        features = self.resnet(imgs)
        features = self.adaptive_pool(features) # (batch_size, 2048, enc_img_size, enc_img_size)
        features = features.permute(0, 2, 3, 1) # (batch_size, enc_img_size, enc_img_size, 2048), enc_img_size=14
        return features

    def fine_tune(self, fine_tune=True): # set some layers to be trainable
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(torch.nn.Module):
    def __init__(self, encoder_dim, lstm_size, attention_dim):
        super(Attention, self).__init__()
        self.encoder_fc = nn.Linear(encoder_dim, attention_dim)
        self.decoder_fc = nn.Linear(lstm_size, attention_dim)
        self.att_fc = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        

    
    def forward(self, features, hidden_state):
        '''
            features.shape: (batch_size, num_pixels, encoder_dim)
            hidden_state: (batch_size, lstm_size)

        '''
        att1 = self.encoder_fc(features) # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_fc(hidden_state) # (batch_size, attention_dim)
        
        att = self.att_fc(self.relu(att1 + att2.unsqueeze(1))).squeeze(2) # (batch_size, num_pixels)
        alpha = self.softmax(att) # (batch_size, num_pixels)

        attention_weighted_encoding = (alpha.unsqueeze(2) * features).sum(dim=1) # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class Decoder(torch.nn.Module):
    def __init__(self, attention_dim, embedding_size, lstm_size, vocab_size, input_dim): # input_dim is not 2048 if noise exists
        super(Decoder, self).__init__()

        self.attention_dim = attention_dim
        self.input_dim = input_dim
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.rnn_cell = nn.LSTMCell(self.input_dim + embedding_size, lstm_size, bias=True)
        self.h_fc = nn.Linear(self.input_dim, lstm_size)
        self.c_fc = nn.Linear(self.input_dim, lstm_size)
        self.f_beta = nn.Linear(lstm_size, self.input_dim)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()


        self.classifier = nn.Linear(lstm_size, vocab_size)
        self.attention = Attention(input_dim, lstm_size, attention_dim)


    def forward(self, features, captions, lengths):
        '''
            features: (batch_size, enc_image_size, enc_image_size, input_dim)
            captions: (batch_size, max_length)
            lengths: (batch_size, )
            
        '''
        # flatten features
        features = features.view(features.size(0), -1, features.size(-1)) # (batch_size, num_pixels, input_dim)
        
        # embedding
        embeddings = self.embeddings(captions) # (batch_size, max_length, embedding_size)

        # initialize LSTM states
        mean_features = features.mean(dim=1) # (batch_size, input_dim)
        hidden_state = self.h_fc(mean_features) # (batch_size, lstm_size)
        cell_state = self.c_fc(mean_features) # (batch_size, lstm_size)


        #############################################
        # Run LSTM
        #############################################
        decoder_lengths = [length - 1 for length in lengths]

        batch_size = features.size(0)
        num_pixels = features.size(1)
        y_predicted = torch.zeros(batch_size, max(decoder_lengths), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decoder_lengths), num_pixels).to(device)
        
        for step in range(max(decoder_lengths)):
            curr_batch_size = sum([l > step for l in decoder_lengths])

            attention_weighted_encoding, alpha = self.attention(features[:curr_batch_size], hidden_state[:curr_batch_size]) # (curr_batch_size, input_dim)

            gate = self.sigmoid(self.f_beta(hidden_state[:curr_batch_size])) # (curr_batch_size, input_dim)
        
            attention_weighted_encoding = gate * attention_weighted_encoding # (curr_batch_size, input_dim)
            hidden_state, cell_state = self.rnn_cell(torch.cat([embeddings[:curr_batch_size, step, :], attention_weighted_encoding], dim=1), (hidden_state[:curr_batch_size], cell_state[:curr_batch_size]))
            y_pred = self.classifier(self.dropout(hidden_state))
            y_predicted[:curr_batch_size, step, :] = y_pred
            alphas[:curr_batch_size, step, :] = alpha

        return y_predicted, captions, decoder_lengths, alphas

    def inference2(self, features, pre_input, max_length=30, pred_mode='max'):
        '''
            Input:
                features: (batch_size, num_pixels, input_dim)
                pre_input: (batch_size, pre_length, ): list. e.g., [sos_idx] / [sos_idx, xx_idx, ...]
                pred_mode: Mode of predicting the next work from the output of LSTM cell. Default value is 'max', which means take the word with the max probability. Whereas 'prob' chooses the word based on the distribution
            Output:
                captions: (batch_size, max_length)
        '''
        max_length -= 1 # adjust
        pre_length = pre_input.shape[1]

        # embedding
        #embeddings = self.embeddings(torch.LongTensor(pre_input).to(device)) # (batch_size, pre_length, embedding_size)
        embeddings = self.embeddings(pre_input) # (batch_size, pre_length, embedding_size)

        # initialize LSTM states
        mean_features = features.mean(dim=1) # (batch_size, input_dim)
        hidden_state = self.h_fc(mean_features) # (batch_size, lstm_size)
        cell_state = self.c_fc(mean_features) # (batch_size, lstm_size)

        #############################################
        # Run LSTM
        #############################################
        batch_size = features.size(0)
        num_pixels = features.size(1)
        #captions = list()
        captions = torch.zeros((batch_size, max_length)).long().to(device)
        inputs = embeddings[:, 0, :]
        for step in range(max_length):
            # get attention
            awe, alpha = self.attention(features, hidden_state) # (batch_size, input_dim)
            gate = self.sigmoid(self.f_beta(hidden_state)) # (batch_size, input_dim)
            awe = gate * awe # (batch_size, input_dim)
            hidden_state, cell_state = self.rnn_cell(torch.cat([inputs, awe], dim=1), (hidden_state, cell_state))
            y_pred = self.classifier(hidden_state)

            #_, y_pred = y_pred.max(1) # y_pred: (batch_size, )
            #captions[:, step] = y_pred
            #captions.append(y_pred)
            if step + 1 >= pre_length:
                if pred_mode == 'prob':
                    y_pred = torch.multinomial(torch.softmax(y_pred, dim=1), 1).squeeze(1) # (batch_size, )
                else:
                    _, y_pred = y_pred.max(1) # y_pred: (batch_size, )
                #captions[:, step] = pre_input[:, step]
                captions[:, step] = y_pred
                inputs = self.embeddings(y_pred)
            else:
                captions[:, step] = pre_input[:, step + 1]
                #inputs = embeddings[:, step, :]
                inputs = embeddings[:, step + 1, :]
        #captions = torch.stack(captions, 1)

        return captions



    def inference_beamsearch(self, features, sos_idx, eos_idx, beam_size=20, max_length=30, all_captions=False):
        '''
            features: (1, enc_image_size, enc_image_size, input_dim)
            sos_idx: index of <sos>
            beam_size: 20 based on the paper "Show and Tell: A Neural Image Caption Generator"
        '''
        max_length -= 1 # adjust
        # flatten features
        features = features.view(1, -1, features.size(-1)) # (1, num_pixels, input_dim)
        batch_size = features.size(0)
        num_pixels = features.size(1)
        features = features.expand(beam_size, num_pixels, features.size(-1)) # (beam_size, num_pixels, input_dim)
        
        # embedding
        curr_indices = torch.Tensor([sos_idx] * beam_size).long() # (beam_size,)
        inputs = self.embeddings(torch.LongTensor(curr_indices).to(device)) # (beam_size, embedding_size)

        # initialize LSTM states
        mean_features = features.mean(dim=1) # (batch_size, input_dim)
        hidden_state = self.h_fc(mean_features) # (batch_size, lstm_size)
        cell_state = self.c_fc(mean_features) # (batch_size, lstm_size)


        #############################################
        # Run LSTM
        #############################################
        captions = list()
        k = beam_size
        captions = torch.LongTensor([[sos_idx]] * beam_size).to(device) # (beam_size, 1)
        scores = torch.zeros(beam_size).float().to(device) # (beam_size, )
        top_k_scores = torch.zeros(k).to(device) # (k). Records the current words' probabilities
        complete_captions = list()
        complete_scores = list()
        for step in range(max_length):
            # get attention
            attention_weighted_encoding, _ = self.attention(features, hidden_state) # (curr_batch_size, input_dim)
            gate = self.sigmoid(self.f_beta(hidden_state)) # (curr_batch_size, input_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding # (curr_batch_size, input_dim)

            hidden_state, cell_state = self.rnn_cell(torch.cat([inputs, attention_weighted_encoding], dim=1), (hidden_state, cell_state))
            y_pred = self.classifier(hidden_state) # (k, vocab_size)

            y_pred = torch.log_softmax(y_pred, dim=1) # softmax => probability; log => convert multiplication to addition
            
            scores = top_k_scores.unsqueeze(1).expand_as(y_pred) + y_pred # (k, vocab_size)

            if step == 0: # There is only <sos> in the word list at the first step, so select scores[0]
                top_k_scores, top_k_indices = scores[0].topk(k, 0, True, True)
            else: # There must be 3 words in the list
                top_k_scores, top_k_indices = scores.view(-1).topk(k, 0, True, True) # (k, ) , (k,) # top_k_indices[0] belongs to [0, k* vocab_size)
            
            curr_indices = top_k_indices % self.vocab_size # since we reshape scores to 1D vector containing k numbers of vocab_size # (k,)
            prev_indices = top_k_indices / self.vocab_size # (k, )
            
            captions = torch.cat([captions[prev_indices.tolist()], curr_indices.unsqueeze(1)], dim=1) # (k, step+1)

            #########################################################
            # Remove the sentences that have reached <eos>
            #########################################################
                
            incomplete_indices = [index for index, word_index in enumerate(curr_indices) if word_index != eos_idx]
            complete_indices = list(set(range(len(curr_indices))) - set(incomplete_indices))
            # update the complete captions
            if len(complete_indices) > 0:
                complete_captions.extend(captions[complete_indices].tolist())
                complete_scores.extend(top_k_scores[complete_indices])
            # select the incomplete sentences
            k -= len(complete_indices)
            if k == 0:
                break

            if step == max_length - 1 and k != 0:
                complete_captions.extend(captions[incomplete_indices].tolist())
                complete_scores.extend(top_k_scores[incomplete_indices])

            curr_indices = curr_indices[incomplete_indices]
            hidden_state = hidden_state[incomplete_indices]
            cell_state = cell_state[incomplete_indices]
            captions = captions[incomplete_indices]
            features = features[incomplete_indices]
            top_k_scores = top_k_scores[incomplete_indices]
                    
            inputs = self.embeddings(curr_indices)


        if all_captions:
            return complete_captions 
        else:
            i = complete_scores.index(max(complete_scores))
            caption = complete_captions[i]

            return caption

            


class Generator(torch.nn.Module):
    def __init__(self, attention_dim, embedding_size, lstm_size, vocab_size, encoder_dim=2048, generator_path = 'data/generator_params.pkl', ad_generator_path='data/ad_generator_params.pkl', load_path=None, noise=False, device='gpu'): # encoder_dim is not 2048 if noise exists
        super(Generator, self).__init__()

        # ------------- constants ----------------
        self.log_every = 10
        self.save_every = 500
        self.save_every_ad = 20
        self.learning_rate = 1e-3
        self.learning_rate_ad = 1e-4
        self.vocab_size = vocab_size


        self.noise = noise
        self.noise_size = 100 if self.noise else 0

        # ------------- encoder ----------------
        fine_tune_encoder = False
        self.encoder = Encoder()
        self.encoder.fine_tune(fine_tune_encoder)

        # ------------- decoder ----------------
        self.decoder = Decoder(attention_dim, embedding_size, lstm_size, vocab_size, encoder_dim + self.noise_size)
        
        # ------------- load model ----------------
        self.generator_path = generator_path
        self.ad_generator_path = ad_generator_path
        if load_path and os.path.exists(load_path):
            print('Start loading %s'%(load_path))
            if device == 'gpu':
                self.load_state_dict(torch.load(load_path))
            else:
                #self.load_state_dict(torch.load(load_path, map_location=lambda stroage, loc: 'cpu'))
                self.load_state_dict(torch.load(load_path, map_location='cpu'))


        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate_ad)

    
    def pre_train(self, dataloader, vocab, alpha_c=1.0):
        '''
            Pre-train discriminator based on data_loader
        '''

        num_steps = len(dataloader)

        for index, (imgs, captions, lengths) in enumerate(dataloader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            features = self.encoder(imgs)
            if self.noise:
                noise = torch.randn(features.shape[0], features.shape[1], features.shape[2], self.noise_size).to(device)
                features = torch.cat([features, noise], dim=3)

            
            y_predicted, captions, lengths, alphas = self.decoder(features, captions, lengths)

            targets = captions[:, 1:]

            y_predicted = pack_padded_sequence(y_predicted, lengths, batch_first=True)[0]
            targets = pack_padded_sequence(targets, lengths, batch_first=True)[0]

            loss = self.loss_fn(y_predicted, targets)
            loss += alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if index % self.log_every  == 0:
                print('Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(index, num_steps, loss.item(), np.exp(loss.item()))) 
    
            if index % self.save_every == 0 and index != 0:
                print('Start saving generator')
                torch.save(self.state_dict(), self.generator_path)

    def translate(self, indices, vocab):
        sentences = list()
        for index in indices:
            word = vocab.idx2word[int(index)]
            sentences.append(word)
            if word == '<eos>':
                break
        return ' '.join(sentences)
 

    def inference(self, vocab, img_path=None, image_dir=None, features=None, translate_flag=False):
        '''
            Generate captions from image path.
            img_path: string. Fullname of the path of the image
            features: (batch_size, enc_img_size, enc_img_size, encoder_dim)
        '''
        # ---------------------- Preprocess images -------------------------------
        if img_path is not None:
            transforms = T.Compose([
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])
            img = Image.open(img_path)
            imgs = transforms(img).to(device).unsqueeze(0)

            with torch.no_grad():
                features = self.encoder(imgs)

        elif features is not None:
            pass
        elif image_dir is not None:
            pass

        else:
            print('ERROR:inference')
            return None


        if self.noise:
            noise = torch.randn(features.shape[0], features.shape[1], features.shape[2], self.noise_size).to(device)
            features = torch.cat([features, noise], dim=3)

        features.view(features.shape[0], -1, features.shape[-1])

        # ---------------------- Generate captions from image features -------------------------------
        #captions = self.sample(features, vocab) # (batch_size, seq_length)
        captions = list()
        with torch.no_grad(): # Avoid accumulating gradients which might result in out of memory
            for feature in features:
                caption = self.decoder.inference_beamsearch(feature.unsqueeze(0), vocab.word2idx['<sos>'], vocab.word2idx['<eos>'], all_captions=False)
                captions.append(caption)
        
        caption = captions[0]

        if translate_flag:
            return self.translate(caption, vocab) # string: <sos> a man ... tree . # no <eos>, but contains <sos>
        else:
            if img_path is not None:
                return caption # list, contains <eos> index
            else:
                return captions


    def estimate_rewards(self, features, features_noise, captions_pred, lengths_pred, vocab, discriminator, num_rollouts=16): 
        '''
            Input:
                ! SORTED BY lengths_pred !
                features: used for discriminator since they are deprived of noises. (batch_size, num_pixels, encoder_dim)
                features_noise: contain noise and thus used for generator: (batch_size, num_pixels, encoder_dim + noise_size)
                lengths_pred: including <sos> & <eos>
            Output:
                reward: (batch_size, decoder_lengths)
        '''
        batch_size = features.shape[0]

        #noise = torch.randn(features.shape[0], features.shape[1], features.shape[2], self.noise_size).to(device)
        #features_noise = torch.cat([features, noise], dim=3)
        #features_noise = features_noise.view(features_noise.size(0), -1, features_noise.size(-1))
        mean_features = features_noise.mean(dim=1) # (batch_size, encoder_dim)
        hidden_state = self.decoder.h_fc(mean_features) # (batch_size, lstm_size)
        cell_state = self.decoder.c_fc(mean_features) # (batch_size, lstm_size)

        decoder_lengths = [length_pred - 1 for length_pred in lengths_pred] # remove <eos>, since the reward at the position of <eos> does not have corresponding log_action to multiply

        rewards = torch.zeros(batch_size, max(decoder_lengths)).to(device)

        for i in range(num_rollouts):
            for step in range(max(decoder_lengths)):
                curr_batch_size = sum([l > step for l in decoder_lengths])
                # get current incomplete captions
                inputs = captions_pred[:curr_batch_size, :step + 1] # (curr_batch_size, step + 1)
                
                # infer complete captions from current partial captions
                captions_step = self.decoder.inference2(features_noise[:curr_batch_size], inputs, pred_mode='prob') # (curr_batch_size, max_length=30)

                # compute lengths_step
                lengths_step = list() # (curr_batch_size)
                for caption_step in captions_step:
                    l = list(caption_step.cpu().numpy())
                    if vocab.word2idx['<eos>'] in l:
                        eos_pos = list(caption_step.cpu().numpy()).index(vocab.word2idx['<eos>']) # find pos of <eos>
                    else:
                        eos_pos = len(l) - 1
                    lengths_step.append(eos_pos + 1)

                # calculate the probability of being true for the predicted complete sentences as the reward for current timestep
                with torch.no_grad():
                    rewards[:curr_batch_size, step] = discriminator.predict(features[:curr_batch_size], captions_step, lengths_step, device) # predict needs captions that contain <eos>
        rewards /= num_rollouts

        return rewards # (batch_size, max(decoder_lengths))
            
    def ad_train(self, dataloader, discriminator, vocab, gamma=2.0, update_every=20, num_rollouts=16, num_batches=None, alpha_c=1.0):
        '''
            Input:
                gamma: reduce the reward
                update_every: the intervals between updates
        '''
        num_steps = len(dataloader)

        for i, (imgs, _1, _2) in enumerate(dataloader):
            imgs = imgs.to(device)

            with torch.no_grad():
                features = self.encoder(imgs)

            batch_size = features.size(0)

            #------------------------ Predict Captions -------------------------
            # Perhaps we should not use beamsearch???
            captions_pred = self.inference(vocab, features=features) # self.inference contains noise addition

            if self.noise:
                noise = torch.randn(features.shape[0], features.shape[1], features.shape[2], self.noise_size).to(device)
                features_noise = torch.cat([features, noise], dim=3)
            features_noise = features_noise.view(features_noise.size(0), -1, features_noise.size(-1)) # (batch_size, num_pixels, encoder_dim)

            # sort features_noise and captions_pred based on the length of captions_pred
            sorted_indices, captions_pred = zip(*sorted(enumerate(captions_pred), key=lambda x: len(x[1]), reverse=True))
            sorted_indices = list(sorted_indices)
            captions_pred = list(captions_pred)
            features_noise = features_noise[list(sorted_indices)]

            lengths_pred = [len(caption_pred) for caption_pred in captions_pred]
            
            max_length_pred = len(captions_pred[0])
            for index, caption_pred in enumerate(captions_pred):
                captions_pred[index] = caption_pred + [0] * (max_length_pred - len(caption_pred))
            captions_pred = torch.LongTensor(captions_pred).to(device)

            #------------------------ Estimate Rewards -------------------------
            #rewards = discriminator.predict(features_noise, captions_pred, lengths_pred, device) # (batch_size,)

            #------------------------ Initialize states for Attention -------------------------
            mean_features_noise = features_noise.mean(dim=1) # (batch_size, encoder_dim)
            hidden_state = self.decoder.h_fc(mean_features_noise) # (batch_size, lstm_size)
            cell_state = self.decoder.c_fc(mean_features_noise) # (batch_size, lstm_size)

            # ------------------- Run LSTM --------------------------
            decoder_lengths = [length_pred - 1 for length_pred in lengths_pred] # remove <eos>

            y_predicted = torch.zeros(batch_size, max(decoder_lengths), self.vocab_size).to(device)
            actions = torch.zeros(batch_size, max(decoder_lengths)).long().to(device)
            inputs = self.decoder.embeddings(torch.LongTensor([vocab.word2idx['<sos>']] * batch_size).to(device)) # (batch_size, embedding_size)
            
            for step in range(max(decoder_lengths)):
                curr_batch_size = sum([l > step for l in decoder_lengths])

                # get attention_weighted_encoding
                awe, _ = self.decoder.attention(features_noise[:curr_batch_size], hidden_state[:curr_batch_size]) # (curr_batch_size, encoder_dim)
                gate = self.decoder.sigmoid(self.decoder.f_beta(hidden_state[:curr_batch_size])) # (curr_batch_size, encoder_dim)
                awe = gate * awe # (curr_batch_size, encoder_dim)

                # run rnn cell
                hidden_state, cell_state = self.decoder.rnn_cell(torch.cat([inputs[:curr_batch_size, :], awe], dim=1), (hidden_state[:curr_batch_size], cell_state[:curr_batch_size]))
                
                y_pred = self.decoder.classifier(self.decoder.dropout(hidden_state)) # (curr_batch_size, vocab_size)
                y_predicted[:curr_batch_size, step, :] = y_pred
                actions[:curr_batch_size, step] = y_pred.max(1)[1] # Suppose the vocab with max prob is the correct action

                inputs = torch.multinomial(torch.softmax(y_pred, dim=1), 1).squeeze(1) # (curr_batch_size, )
                inputs = self.decoder.embeddings(inputs) # (curr_batch_size, embedding_size)

            # y_predicted: (batch_size, max_length, vocab_size)
            
            y_predicted = F.log_softmax(y_predicted, dim=2) # (batch_size, max_decoder_length, vocab_size)

            ad_loss = 0
            features= features.view(features.size(0), -1, features.size(-1)) # (batch_size, num_pixels, encoder_dim)
            rewards = self.estimate_rewards(features, features_noise, captions_pred, lengths_pred, vocab, discriminator, num_rollouts) # (batch_size, decoder_lengths)
            
            for index in range(batch_size):
                batch_loss = 0.0 # loss of the current batch
                for timestep in range(decoder_lengths[index]):
                    curr_idx = actions[index][timestep]
                    log_prob = y_predicted[index][timestep][curr_idx] # log probability of curr index/word. Note that log_softmax has already been called
                    #ad_loss += -y_predicted[index][timestep][actions[index][timestep]] * (rewards[index][timestep])
                    reward = rewards[index][timestep] 
                    # ~~~ I think the rewards in front of the sentence should not be too high, since they have less affect in the future ~~~
                    #reward = reward * (1.0 / (gamma ** (decoder_lengths[index] - timestep)))

                    batch_loss += - log_prob * reward # Policy Gradient
                #ad_loss += batch_loss / decoder_lengths[index]
                ad_loss += batch_loss # According to the formula, the loss is the summation rather than mean

            #print('ad_loss(before division):', ad_loss)
                
            ad_loss /= batch_size
            #print('ad_loss(final):', ad_loss)
            
            self.optimizer.zero_grad()
            ad_loss.backward()
            if (i + 1) % self.log_every  == 0:
                print('Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f} ---ad'.format(i + 1, num_steps, ad_loss.item(), np.exp(ad_loss.item()))) 
    
            if (i + 1) % self.save_every_ad == 0:
                print('Start saving ad_generator')
                ad_generator_path = 'data/pkl/ad_generator_params.pkl'
                torch.save(self.state_dict(), ad_generator_path)

                if (i + 1) % 100 == 0:
                    print('Start saving ad_generator %d'%(i + 1))
                    ad_generator_path = 'data/pkl/ad_generator_params_%d.pkl'%(i + 1)
                    torch.save(self.state_dict(), ad_generator_path)

            if (i + 1) % update_every == 0: # !!! Do not update generator every batch, since it does not conform to Monte Carlo's requirements which requires a sufficient number of samples
                for param in self.parameters():
                    if param.requires_grad == True:
                        param.grad /= update_every
                print('Start updating')
                self.optimizer.step()


            if num_batches and i + 1 >= num_batches:
                break

