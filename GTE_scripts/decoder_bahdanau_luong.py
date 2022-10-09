class DecoderRNN(nn.Module):
    
    def __init__(self, embedding_matrix_size, hidden_size, 
                 num_layers, attn_type="luong", attn_func="mlp", dropout=0.0):
        super().__init__()
        self.embedding_size = embedding_matrix_size[1]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = embedding_matrix_size[0]
  
        self.attn = GlobalAttention(hidden_size, attn_func)
    
        # Build the RNN.
        self.rnn = nn.LSTM(input_size=self._input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first = True,
                           dropout=dropout)
        
        if attn_type == "luong":
            self.proj = self._attn_proj
        self.out = nn.Linear(hidden_size, self.num_classes)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, emb_hypo, enc_out, enc_state, mask_prem, 
                **kwargs):
        """
        Args:
            emb_hypo (FloatTensor): embedded hypothesis sequence
                 ``(batch, hypo_len, embed_dim)``.
            enc_out (FloatTensor): hidden vectors from the last encoder layer
                 ``(batch, prem_len, hidden)``.
            enc_state tuple(FloatTensor, FloatTensor): hidden and cell state encoder
                 ``(num_layers, batch_size, hidden)``.
            mask_prem (BoolTensor): the encoder sequence mask
                ``(batch, prem_len)``.
        Returns:
            (FloatTensor, (FloatTensor, FloatTensor), FloatTensor):
            * dec_outs: output from the decoder (after attn)
              ``( batch, hypo_len, hidden)``.
            * dec_state: hidden and cell state from decoder
              ``(num_layers, batch, hidden)``.
            * attns: distribution over premise at each hypo token
              ``( batch, hypo_len, prem_len)``.
        """

        dec_outs, dec_state, attns = self._run_forward_pass(
            emb_hypo, enc_out, enc_state, mask_prem, **kwargs)
        
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs, dim=1)

        if type(attns) == list:
            attns = torch.stack(attns, dim=1)
            
        return dec_outs, dec_state, attns




class DecoderLuong(DecoderRNN):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    
    Based around the approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`    
    """
    
    def _run_forward_pass(self, emb_hypo, enc_out, enc_state, mask_prem, hypo_len=None):
        
        """
        See RNNDecoder.forward() for description
        of arguments and return values.
        """
        if hypo_len is None: 
            hypo_len = torch.tensor([1])#.to(device)
            
        hypo_packed = nn.utils.rnn.pack_padded_sequence(emb_hypo,
                                                        hypo_len.cpu(),
                                                        enforce_sorted = False,
                                                        batch_first = True)
        
        # lstm_state shape: [num_layers, batch_size, hidden_size] 
        # lstm_output shape: [batch_size, hypo_len, hidden_size] 
        lstm_out, lstm_state = self.rnn(hypo_packed, enc_state)         
        lstm_out = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first = True)[0] 
        
        # attention_weights shape: [batch_size, hypo_len, prem_len]
        # context_vecs shape: [batch_size, hypo_len, hidden_size]
        attention, c_t = self.attn(lstm_out, enc_out, mask_prem)
        
        # concatenate
        concat_c = torch.cat([c_t, lstm_out], dim=-1)
        attn_h = torch.tanh(self.proj(concat_c))
        attn_h = self.dropout(attn_h) # attn_h shape: [batch_size, hypo_len, hidden_size]
        prediction = F.log_softmax(self.out(attn_h), dim=-1)
        
        return prediction, lstm_state, attention
    
    @property
    def _attn_proj(self):
        """ Add projection layer when concating hidden with attention vectors."""
        return nn.Linear(self.hidden_size*2, self.hidden_size)
    
    @property
    def _input_size(self):
        """ Specify a different input size if using input feeding approach."""
        return self.embedding_size




class DecoderBahdanau(DecoderRNN):
    """  
    RNN decoder with attention based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`
    """
    
    def _run_forward_pass(self, emb_hypo, enc_out, enc_state, mask_prem, **kwargs):
        
        """
        See RNNDecoder.forward() for description
        of arguments and return values.
        """
        
        dec_outs, attns = [], []
        dec_state = enc_state
        
        for token_t in emb_hypo.split(1, dim=1):
            
            # Check
            hypo_batch, hypo_len, dim = token_t.size()
            prem_batch, prem_len, dim = enc_out.size()
            
            hidden, cell = dec_state
            
            # compute attention with the hidden from the last encoder layer
            h_tm1 = hidden[-1] # (batch_size, hidden_size)
            attention, c_t = self.attn(h_tm1, enc_out, mask_prem)
            
            # replace final layer cell with context vector
            assert cell[-1].size() == c_t.squeeze(1).size()
            cell[-1] = c_t.squeeze(1)
            
            # alternative solutions:
            # concat c_t and token_t and amplify the dim to embed_size + hidden_size 
            # input_decoder = torch.cat([c_t, token_t], 2)
            # h_t, dec_state = self.rnn(input_decoder, (hidden, cell))
            
            # concat c_t and hidden and input as hidden and project with mlp
            # concat_c = torch.cat([c_t, h_tm1], 2).view(batch * hypo_len, dim*2)
            # attn_h = self.linear_out(concat_c).view(batch, hypo_len, dim)
            # if self.attn_type in ["general", "dot"]:
            #    attn_h = torch.tanh(attn_h)
            
            # get the hidden at step t and compute prediction
            h_t, dec_state = self.rnn(token_t, (hidden, cell))
            prediction = F.log_softmax(self.out(h_t).squeeze(1), dim=-1)
            
            attns.append(attention)
            dec_outs.append(prediction)
            
        return dec_outs, dec_state, attns
    
    @property
    def _input_size(self):
        """Specify a different input size with input feed approach."""
        return self.embedding_size

