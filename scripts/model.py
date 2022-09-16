import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,
                 embed_size,
                 hidden_size,
                 n_layers,
                 dropout,
                 device):
        super().__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.bilstm = nn.LSTM(self.embed_size,
                              self.hidden_size,
                              num_layers = n_layers,
                              bidirectional = True,
                              batch_first = True,
                              dropout = dropout)
        
        self.dropout = nn.Dropout(p = dropout)
        
      
    def forward(self, batch, prem_length):
        # batch : [batch_size, seq_len]
        
        # premise_embedding shape: [batch_size, seq_len, 300]
        premise_embedding = self.dropout(batch)
        
        # Handling padding in Recurrent Networks
        prem_packed = nn.utils.rnn.pack_padded_sequence(premise_embedding,
                                                        prem_length.cpu(),
                                                        enforce_sorted = False,
                                                        batch_first = True)
        
        # lstm_state is a tuple (hidden, cell) with size : [4, batch_size, 256] 
        # lstm_out with size : [batch_size, seq_len, 512]
        lstm_out, lstm_state = self.bilstm(prem_packed)  
        
        # pad lstm output to have same length sequences in batch
        annotations = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)[0] 
        
        # resize from [num_layers * n_directions, batch_size, hidden] to [num_layers, batch_size, hidden * 2]
        hidden = lstm_state[0].reshape(self.n_layers, 2, -1, self.hidden_size)\
            .transpose(1,2).reshape(self.n_layers, -1, 2*self.hidden_size)
        cell = lstm_state[1].reshape(self.n_layers, 2, -1, self.hidden_size)\
            .transpose(1,2).reshape(self.n_layers, -1, 2*self.hidden_size)

        # hidden and cell of size [num_layers, batch_size, hidden] = [2, 32, 512]
        # annotations of size [batch_size, seq_len, 512] 
        return annotations, (hidden, cell)



class AdditiveAttention(nn.Module):
    
    def __init__(self, query_size, key_size, hid_dim, dropout):
        super().__init__()
        self.W_k = nn.Linear(key_size, hid_dim, bias = False)
        self.W_q = nn.Linear(query_size, hid_dim, bias = False)
        self.W_v = nn.Linear(hid_dim, 1, bias = False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_decoder, annotations, mask = None):
        
        query, keys = self.W_q(hidden_decoder), self.W_k(annotations)
        # features (tensor) with shape: (batch_size, nquery, nkeys, 512) 
        features = query.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(self.dropout(features))
        energy = self.W_v(features).squeeze(-1)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention_weights = F.softmax(energy, dim = -1)
        expected_annotation = torch.bmm(attention_weights, annotations)
        
        # attention_weights shape: [batch_size, hypo_len, prem_len]
        # expected_annotation shape: [batch_size, hypo_len, hidden_size]
        return attention_weights, expected_annotation


class Decoder(nn.Module):
    def __init__(self,
                 embedding_matrix_size,
                 hidden_size,
                 n_layers,
                 dropout,
                 device,
                 additive_attention = False
                 ):
        super().__init__()
        self.num_classes = embedding_matrix_size[0]
        self.embed_size = embedding_matrix_size[1]
        self.hidden_size = hidden_size
        self.additive_attention = additive_attention

        self.lstm = nn.LSTM(self.embed_size,
                            self.hidden_size,
                            num_layers = n_layers,
                            batch_first = True, 
                            dropout = dropout)
        
        self.W_a = nn.Linear(self.hidden_size + self.hidden_size , self.hidden_size, bias = True)
        self.W_s = nn.Linear(self.hidden_size, self.num_classes, bias = True)
        self.dropout = nn.Dropout(dropout)
        if additive_attention:
            self.additive_attention_layer = AdditiveAttention(query_size=self.hidden_size,
                                                              key_size=self.hidden_size,
                                                              hid_dim=self.hidden_size//2,
                                                              dropout=dropout)
        
    def dot_attention_layer(self, hidden_decoder, annotations, mask = None):
        # hidden_decoder shape: [batch_size, hypo_len, hidden_size]
        # annotations shape: [batch_size, prem_len, hidden_size]
        # mask shape: [batch_size, 1, prem_len]
        
        # energy shape: [batch_size, hypo_len, prem_len]   
        energy = torch.bmm(hidden_decoder, annotations.permute(0,2,1))
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention_weights = F.softmax(energy, dim = -1)
        expected_annotation = torch.bmm(attention_weights, annotations)
        
        # attention_weights shape: [batch_size, hypo_len, prem_len]
        # expected_annotation shape: [batch_size, hypo_len, hidden_size]
        return attention_weights, expected_annotation
        
    def forward(self, hypothesis_batch, hypo_len, annotations, prev_state, mask = None):
        # hypothesis_batch shape: [batch_size, hypo_len]
        # annotations shape: [batch_size, hypo_len, hidden_size]
        # prev_state shape: ([2, 32, 512], [2, 32, 512])
        # mask(optional) shape: [batch_size, 1, prem_len]
        
        # hypothesis_embedding shape: [batch_size, hypo_len, 300]
        hypothesis_embedding = self.dropout(hypothesis_batch)
        
        # Handling padding in Recurrent Networks
        hypo_packed = nn.utils.rnn.pack_padded_sequence(hypothesis_embedding,
                                                        hypo_len.cpu(),
                                                        enforce_sorted = False,
                                                        batch_first = True)
        
        # lstm_state shape: [num_layers, batch_size, hidden_size] 
        lstm_output, lstm_state = self.lstm(hypo_packed, prev_state) 
        
        # lstm_output shape: [batch_size, hypo_len, hidden_size] 
        lstm_output = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first = True)[0] 
        
        # attention_weights shape: [batch_size, hypo_len, prem_len]
        # expected_annotation shape: [batch_size, hypo_len, hidden_size]
        if self.additive_attention:
            attention_weights, expected_annotation = self.additive_attention_layer(hidden_decoder=lstm_output,
                                                                                        annotations=annotations,
                                                                                        mask=mask)
        else:           
            attention_weights, expected_annotation = self.dot_attention_layer(hidden_decoder=lstm_output,
                                                                                   annotations=annotations, 
                                                                                   mask=mask)
        # attention_hidden shape: [batch_size, hypo_len, hidden_size]
        context_vector = torch.cat((expected_annotation, lstm_output), dim = -1)
        attention_hidden = self.dropout(torch.tanh(self.W_a(context_vector)))
        
        # probabilities shape: [batch_size, hypo_len, vocab_size]
        log_probs = F.log_softmax(self.W_s(attention_hidden), dim = -1)
        return log_probs, lstm_state, attention_weights


class Seq2Seq(nn.Module):
    def __init__(self,
                 pretrained_w2v,
                 hidden_size,
                 n_layers,
                 dropout = 0.2,
                 device = "cpu",
                 additive_attention = False):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding.from_pretrained(embeddings = pretrained_w2v,
                                                      freeze = False,
                                                      padding_idx = 0)
        
        self.encoder = Encoder(pretrained_w2v.size(1),
                               hidden_size//2,
                               n_layers,
                               dropout,
                               device)
        self.decoder = Decoder(pretrained_w2v.size(),
                               hidden_size,
                               n_layers,
                               dropout,
                               device,
                               additive_attention)

    @staticmethod
    def create_mask(seq):    
        # seq shape: [batch_size, seq_len]
        # mask shape: [batch_size, 1, seq_len]
    
        mask = (seq != 0).unsqueeze(1)
        return mask

        
    def forward(self, premise, hypothesis, prem_length, hypo_length, device):
        
        prem_mask = Seq2Seq.create_mask(premise).to(device)
        hypothesis = hypothesis[:,:-1].contiguous()
        
        # Embdedding
        prem_embed = self.embedding(premise)
        hypo_embed = self.embedding(hypothesis)
        
        #Encoding & Decoding
        encoder_output, encoder_state = self.encoder(prem_embed,
                                                     prem_length)
        log_probs, decoder_state, _ = self.decoder(hypo_embed,
                                                   hypo_length,
                                                   encoder_output,
                                                   encoder_state, 
                                                   prem_mask)
        return log_probs
