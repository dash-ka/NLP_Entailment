class GlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.
    
    All models compute the output as
    :math:`c = \sum_{j=1}^{\text{SeqLength}} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    A projection layer can be applied to [quety, context].
    Different ways to compute attention scores:
    * Luong Attention (dot, general):
       * dot: :math:`\text{score}(H_j,q) = H_j^T q`
       * general: :math:`\text{score}(H_j, q) = H_j^T W_a q`
    * Bahdanau Attention (mlp):
       * :math:`\text{score}(H_j, q) = v_a^T \text{tanh}(W_a q + U_a h_j)`
    Args:
       dim (int): dimensionality of query and key
       attn_func (str): type of attention to use, options [dot, general,mlp]
    """

    def __init__(self, dim, attn_func):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        self.attn_func = attn_func
        if self.attn_func == "general":
            self.proj_query = nn.Linear(dim, dim, bias=False)
        elif self.attn_func == "mlp":
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
        
        if self.attn_func in ["general", "dot"]:
            if self.attn_func == "general":
                h_t = self.proj_query(h_t)
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
        one_step = False
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