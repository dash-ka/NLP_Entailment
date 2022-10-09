import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    """
	Args:
	    emb_prem (FloatTensor): premise embedding
            ``(batch_size, seq_len, emb_size)``.
	    prem_len (LongTensor): premise sequence lengths
	    ``(batch_size, )``.

        Returns: (FloatTensor, tuple(FloatTensor, FloatTensor))
	    enc_out : encoder output
            ``(batch_size, seq_len, hidden_size)``.
	    hidden, cell : encoder hidden and cell states
            ``(num_layers, batch_size, hidden_size)``.

    """

    def __init__(self,
                 embed_size,
                 hidden_size,
                 n_layers,
                 dropout):
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
        
      
    def forward(self, emb_prem, prem_len):

        batch_size, seq_len, _ = emb_prem.size()
        
        # Handling padding in Recurrent Networks
        prem_packed = nn.utils.rnn.pack_padded_sequence(emb_prem,
                                                        prem_len.cpu(),
                                                        enforce_sorted = False,
                                                        batch_first = True)
        
        # lstm_state is a tuple (hidden, cell) with size : [4, batch_size, 256] 
        # lstm_out with size : [batch_size, seq_len, 512]
        lstm_out, lstm_state = self.bilstm(prem_packed)  
        
        # pad lstm output to have same length sequences in batch
        enc_out = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)[0] 
        
        # resize from [num_layers*n_dirs, batch_size, hidden] to [num_layers, batch_size, hidden*2]
        hidden = lstm_state[0].reshape(self.n_layers, 2, -1, self.hidden_size)\
            .transpose(1,2).reshape(self.n_layers, -1, 2*self.hidden_size)
        cell = lstm_state[1].reshape(self.n_layers, 2, -1, self.hidden_size)\
            .transpose(1,2).reshape(self.n_layers, -1, 2*self.hidden_size)

        return enc_out, (hidden, cell)