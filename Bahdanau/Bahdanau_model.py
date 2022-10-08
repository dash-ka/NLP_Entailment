class EncoderRNN(nn.Module):
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
        
      
    def forward(self, premise_embedding, prem_length):
        batch_size, seq_len, _ = premise_embedding.size()
        # premise_embedding : [batch_size, seq_len, emb_size]
        
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


class GlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.
    
    All models compute the context vector as
    :math:`c = \sum_{j=1}^{\text{SeqLength}} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then may apply a projection layer to [q, c].

    Differnt ways to compute the attention score:
    * Luong Attention (dot, general):
       * dot: :math:`\text{score}(H_j,q) = H_j^T q`
       * general: :math:`\text{score}(H_j, q) = H_j^T W_a q`
    * Bahdanau Attention (additive):
       * :math:`\text{score}(H_j, q) = v_a^T \text{tanh}(W_a q + U_a h_j)`

    Args:
       dim (int): dimensionality of query and key
       attn_type (str): type of attention to use, options [dot, additive]
    """

    def __init__(self, dim, attn_type):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_q = nn.Linear(dim, dim, bias=True)
        self.W_v = nn.Linear(dim, 1, bias=False)

    def score(self, h_t, annotations):
        """
        Args:
          h_t (FloatTensor): sequence of queries ``(batch, hypo_len, dim)``
          annotations (FloatTensor): sequence encoder outs ``(batch, prem_len, dim)``
        Returns:
          FloatTensor: raw attention scores (unnormalized) for each premise index
            ``(batch, hypo_len, prem_len)``
        """
        # Check input sizes
        batch_hypo, len_hypo, dim = h_t.size()
        batch_prem, len_prem, _ = annotations.size()

        if self.attn_type == "dot":
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            annotations = annotations.transpose(1, 2).contiguous()
            return torch.bmm(h_t, annotations)
        else:
            query = self.W_q(h_t).unsqueeze(2)
            keys = self.W_k(annotations).unsqueeze(1)
            align_scores = torch.tanh(query + keys)
            assert align_scores.size() == (batch_hypo, len_hypo, len_prem, dim)
            return self.W_v(align_scores).squeeze(-1)

    def forward(self, hidden_decoder, enc_outputs, mask_prem=None):
        """
        Args:
          hidden_decoder (FloatTensor): query vectors ``(batch, hypo_len, dim)``
          enc_outputs (FloatTensor): source vectors ``(batch, prem_len, dim)``
          mask_prem (BoolTensor): the source context lengths ``(batch, prem_len)``
        Returns:
          (FloatTensor, FloatTensor):
          * Context vector c_t ``(batch, hypo_len, dim)``
          * Attention distribtutions for each query alpha ``(batch, (?), prem_len)``
        """

        # one step input
        if hidden_decoder.dim() == 2:
            one_step = True
            hidden_decoder = hidden_decoder.unsqueeze(1) # to 3D

        batch_size, prem_len, dim = enc_outputs.size()
        batch_size, hypo_len, dim = hidden_decoder.size()
        
        # compute attention scores
        align = self.score(hidden_decoder, enc_outputs)
        if mask_prem is not None:
            mask_prem = mask_prem.unsqueeze(1).expand(align.size())
            align.masked_fill_(~mask_prem, -float("inf"))
        alpha = F.softmax(align, dim=-1) #(batch_size, hypo_len, prem_len)
        # each context vector c_t is the weighted average
        # over all the source hidden states
        c_t = torch.bmm(alpha, enc_outputs)

        if one_step:
            alpha = alpha.squeeze(1)

        return  alpha, c_t



class DecoderRNN(nn.Module):
    
    def __init__(self, embedding_matrix_size, hidden_size, 
                 num_layers, dropout=0.0):
        super().__init__()
        self.embedding_size = embedding_matrix_size[1]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = embedding_matrix_size[0]
  
        self.attn = GlobalAttention(hidden_size, "additive")
        
        # Build the RNN.
        self.rnn = nn.LSTM(input_size=self._input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first = True,
                           dropout=dropout)
        
        self.out = nn.Linear(hidden_size, self.num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, token_t, dec_state, enc_out, enc_state, mask_prem, 
                **kwargs):
        """
        Args:
            token_t (FloatTensor): hypothesis embedded tokens
                 ``(batch, 1, embed_dim)``.
            dec_state (FloatTensor, FloatTensor): hidden, cell state decoder
                 ``(num_layers, batch_size, hidden)``.
            enc_out (FloatTensor): vectors from the encoder
                 ``(batch, prem_len, hidden)``.
            enc_state (FloatTensor, FloatTensor): hidden, cell state encoder
                 ``(num_layers, batch_size, hidden)``.
            mask_prem (BoolTensor): the encoder mask
                ``(batch, prem_len)``.
        Returns:
            (FloatTensor, (FloatTensor, FloatTensor), FloatTensor):
            * prediction: output from the decoder (after attn)
              ``( batch, hypo_len, hidden)``.
            * dec_state: hidden and cell state from decoder
              ``(num_layers, batch, hidden)``.
            * attns: distribution over premise at each hypo token
              ``( batch, hypo_len, prem_len)``.
        """

        # Check
        hypo_batch, hypo_len, dim = token_t.size()
        prem_batch, prem_len, dim = enc_out.size()
        
        hidden, cell = dec_state
        # take only the hidden from the last encoder layer  
        h_tm1 = hidden[-1] # (batch_size, hidden_size)
        
        # compute attention
        attention, c_t = self.attn(h_tm1, enc_out, mask_prem)
        assert cell[-1].size() == c_t.squeeze(1).size()
        cell[-1] = c_t.squeeze(1)
        
        #input_decoder = torch.cat([c_t, token_t], 2) 
        # assuming you added <bos> in preprocessing
        h_t, dec_state = self.rnn(token_t, (hidden, cell))
        prediction = F.log_softmax(self.out(h_t).squeeze(1), dim=-1)
        
        return prediction, dec_state, attention

    @property
    def _input_size(self):
        """Using input feed by concatenating input with attention vectors."""
        return self.embedding_size 



class Seq2Seq(nn.Module):
    
    """Args:
      pretrained_Word2Vec
      hidden_size: size of the hidden state
      num_layers: number of layers in encoder and decoder
      dropout: dropout fraction
    """
    def __init__(self,
                 pretrained_w2v,
                 hidden_size,
                 n_layers,
                 dropout = 0.2):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings = pretrained_w2v,
                                                      freeze = False,
                                                      padding_idx = 0)
        
        self.encoder = EncoderRNN(pretrained_w2v.size(1),
                               hidden_size//2,
                               n_layers,
                               dropout)
        self.decoder = DecoderRNN(pretrained_w2v.size(),
                               hidden_size,
                               n_layers,
                               dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, prem, hypo, prem_lengths):
            
        hypo = hypo[:,:-1].contiguous()
        mask_prem = torch.ne(prem, .0)
        
        emb_prem = self.embedding(prem)
        emb_hypo = self.embedding(hypo)
        
        emb_prem = self.dropout(emb_prem)
        emb_hypo = self.dropout(emb_hypo)
        
        enc_out, enc_state = self.encoder(emb_prem, prem_lengths)
        
        dec_outs, attns = [], []
        dec_state = enc_state
        
        for token_t in emb_hypo.split(1, dim=1):
            dec_out, dec_state, attn =\
            self.decoder(token_t, dec_state, enc_out, enc_state, mask_prem)
            attns.append(attn)
            dec_outs.append(dec_out)
            
        attns = torch.stack(attns, dim=1)
        dec_outs = torch.stack(dec_outs, dim=1)
        
        return dec_outs, attns