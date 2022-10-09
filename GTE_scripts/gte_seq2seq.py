import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2Seq(nn.Module):
    
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
        self.decoder = DecoderLuong(pretrained_w2v.size(),
                               hidden_size,
                               n_layers,
                               attn_type="luong",
                               attn_func="dot",
                               dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, prem, hypo, prem_lengths, hypo_lengths):
        
        """Args:
               prem (LongTensor): premise sequence
                 ``(batch, prem_len)``.
               hypo (LongTensor): hypothesis sequence
                 ``(batch, prem_len)``.
               prem_lengths (LongTensor): premise sequence length
                 ``(batch,)``.
               hypo_lengths (LongTensor): hypo sequence length
                 ``(batch,)``.
        """
            
        hypo = hypo[:,:-1].contiguous()
        mask_prem = torch.ne(prem, .0)
        
        emb_prem = self.embedding(prem)
        emb_hypo = self.embedding(hypo)
        
        emb_prem = self.dropout(emb_prem)
        emb_hypo = self.dropout(emb_hypo)
        
        enc_out, enc_state = self.encoder(emb_prem, prem_lengths)
        dec_out, dec_state, attention = self.decoder(emb_hypo, enc_out, enc_state, mask_prem, hypo_len=hypo_lengths)
        
        return dec_out, attention
