import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import EncoderRNN
from decoder import *

import pandas as pd
import re, time, copy
from string import punctuation
from tqdm.notebook import tqdm
from collections import OrderedDict

from torchtext.vocab import vocab, build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader



class JointDataset(Dataset):
    
    def __init__(self, data):
        self.df = data
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        pair = self.df.iloc[idx]
        return pair[0], pair[1]

    @classmethod
    def load_dataset(cls, paths):
        """Load and join MultiSNLI/SNLI datasets. 
           NOTE: files are expected to have .txt format !
        
        Args:
        paths (List[str]):a list of paths to the data to load
        
        Returns:
        an instance of Dataset
        """
        datasets = []
        
        for path in paths:
            
            # load Multi genre NLI dataset
            if "multinli" in path:
                mnli = pd.read_table(path, delimiter ="\t",on_bad_lines='skip', index_col = "gold_label")\
                 .loc[["entailment"], ["sentence1", "sentence2", "genre"]].dropna()
                # filter out examples from "telephone" and erroneous genre  
                mnli = mnli.loc[~ mnli["genre"].isin(["contradiction", "telephone"])]
                datasets.append(mnli[["sentence1", "sentence2"]])
                
            # Load SNLI dataset
            if "snli" in path:
                datasets.append(pd.read_table(path, delimiter ="\t", index_col = "gold_label")\
                 .loc[["entailment"], ["sentence1", "sentence2"]].dropna())
            
        data = pd.concat(datasets, ignore_index=True)
       
        return cls(data)
    
    
    
class Vocabulary:
    
    def __init__(self, vocabulary, pretrained_vectors):
        self.vocabulary = vocabulary
        self.pretrained = pretrained_vectors
    
    
    @staticmethod 
    def yield_tokens(data_iterator):
        """ Generator yields one list of tokens from one premise-hypothesis pair. 
        
        Args:
        train_data (str, str): training dataset
        
        Returns:
        list of tokens
        
        """
   
        for row in data_iterator:
            yield word_tokenizer(" ".join(row))
            
            
    @staticmethod
    def get_n_freq_w2v(wv, n_most_freq=20000):
        
        """Extract n_most_freq words from pretrained Word2Vec model
        
        Args:
        wv: pretrained Word2Vec model
        n_most_freq(int): number of words to extract from the model
        
        Returns:
        freq_words (list): n most frequent words from Word2Vec
        word_vectors (torch.FloatTensor) with shape (n_most_freq, 300)
        """ 
        regex = "[^a-zA-Z0-9']+"
        pattern = re.compile(regex)
        
        freq_words = ["<pad>"]
        word_vectors = [torch.zeros(300)]
        
        print(f"Collecting representation for {n_most_freq} most freqwords...")
        
        for w in tqdm(wv.index2entity):
            word = w.lower() 
            if len(freq_words) < n_most_freq:
                if (not re.search(pattern, word)) and (word not in freq_words):
                    try :
                        word_vectors.append(torch.tensor(wv[word]))
                        freq_words.append(word)
                    except KeyError:
                        continue
            else:
                break
                
        return freq_words, torch.stack(word_vectors)
    
    
    @classmethod
    def build_vocabulary(cls, data, wv=None, min_freq=1, n_most_freq =10000):
        """Extract vocabulary from training data
        
        Args:
        data (str, str): sentence_pairs
        wv: pretrained Word2Vec model
        min_freq(int): minimum word frequency to incude the word in vocabulary
        n_most_freq(int): number of words to extract from pretrained Word2Vec model
        
        Returns:
        An instance of Vocabulary class
        """
        
        # build the vocabulary mapping from the training data
        vocabulary = build_vocab_from_iterator(cls.yield_tokens(multi_train), min_freq=2)
        
        # if the pretrained model is provided extract n_most_frequent words
        if wv is not None:
            freq_words, word_vectors = cls.get_n_freq_w2v(wv, n_most_freq)
            pretrained = vocab(OrderedDict([(token, 1) for token in freq_words]),
                              specials = ["<unk>", "<sos>", "<eos>"], special_first=False)
            pretrained.set_default_index(pretrained["<unk>"])
            
            # extend vocabulary with n most frequent words with pretrained vectors
            idx = len(pretrained)
            for token in vocabulary.get_stoi():
                if pretrained[token] == pretrained["<unk>"]: #50000
                    pretrained.insert_token(token,  idx) 
                    idx += 1
            vocabulary = pretrained
            
        return cls(vocabulary, word_vectors) # return an instance of the class
    
    

    def sentence2tensor(self, sentence):
        """ 
        Encodes tokens with indices;
        appends <EOS> token;
        transforms into a torch.tensor.
        
        Args:
        sentence (str): row string 
        
        Returns:
        vectorized sentence
        """
        indices = [self.vocabulary[word] for word in word_tokenizer(sentence)]
        indices.append(self.vocabulary["<eos>"])
        return torch.tensor(indices)
    
    
    
    
def word_tokenizer(text):
    """ Minimal sentence cleaning and tokenization."""
    
    tokenizer = get_tokenizer("basic_english")
    text = re.sub("-", " ", text)
    tokens = [w if w in punctuation else re.sub("[" + punctuation + "]", "", w.lower()) for w in tokenizer(text)]
    return tokens




class Seq2Seq(nn.Module):
    
    """
    Seq2Seq model is a container class for encoder and decoder blocks.
    It internally implements the sentence embedding and performs a forward pass.
    The ``embed_sentence`` function extracts the embeddings from 2 embedding layers.
    ``pretrained_embedding`` is a look_up table for pretrained w2v vectors. It remains freezed during training,
    while the ``trainable_embedding`` holds embeddings for vocabulary terms without a pretrained representation.
    The latter will be learned during training.
    
    Args:
        oov_size(int): number of words in vocabulary without pretrained w2v vector
        vocab_size(int): total number of word in vocabulary
        pretrained_w2v(FloatTensor): embedding matrix with pretrained vector representations
        ``(num_pretrained_words, 300)``
        hidden_size(int): dimention of enc/dec hidden representation
        n_layers: number of layers in encoder and decoder
        device: device
        attn_type(str): either "luong" or "bahdanau" attention architecture
        attn_func(str): a string telling how attention score is computed : 
         - "mlp" = multilayer perceptron
         - "general" = projects query vector with a linear layer
         - "dot" = simple dot product between query and keys
         dropout(float): dropout fraction
       
    """
    
    def __init__(self, oov_size, vocab_size,
                 pretrained_w2v,
                 hidden_size,
                 n_layers,
                 device,
                 attn_type="luong",
                 attn_func="dot",
                 dropout = 0.2):
        super().__init__()

        self.pretrained_embedding = nn.Embedding.from_pretrained(embeddings = pretrained_w2v,
                                                      freeze = True, padding_idx=0)
        
        self.trainable_embedding = torch.nn.Embedding(oov_size, pretrained_w2v.size(1))
        
        self.encoder = EncoderRNN(pretrained_w2v.size(1),
                               hidden_size//2,
                               n_layers,
                               dropout)
        if attn_type == "luong":
            self.decoder = DecoderLuong(
                embedding_matrix_size=(vocab_size, pretrained_w2v.size(1)),
                               hidden_size=hidden_size,
                               num_layers=n_layers,
                               attn_type=attn_type,
                               attn_func=attn_func,
                               dropout=dropout)
        else:
            self.decoder = DecoderBahdanau((vocab_size, pretrained_w2v.size(1)),
                               hidden_size=hidden_size,
                               num_layers=n_layers,
                               attn_type=attn_type,
                               attn_func=attn_func,
                               dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        
        
        
    def embed_sentence(self, sentence):
        
        """ 
        Embed the sentece using both pretrained and trainable Embedding layers.
        
        Args:
        sentence (LongTensor): a tensor of indices
        
        Returns:
        emb_sent (FloatTensor) : embedded sentence
        """
        embedding_mask = (sentence >= self.pretrained_embedding.num_embeddings).to(self.device)
        pretrained_embed = sentence.clone().to(self.device)
        pretrained_embed[embedding_mask] = 0
        emb_sent = self.pretrained_embedding(pretrained_embed)
    
        # rescale tokens without representation
        sentence -= self.pretrained_embedding.num_embeddings
        
        # zero out tokens with pretrained embedding
        sentence[~embedding_mask] = 0
        non_pretrained_embed = self.trainable_embedding(sentence)
    
        # change tokens from placeholder embedding into trainable embeddings.
        emb_sent[embedding_mask] = non_pretrained_embed[embedding_mask]
        
        return emb_sent 
        
        
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
        
        emb_prem = self.embed_sentence(prem)
        emb_hypo = self.embed_sentence(hypo)
        
        emb_prem = self.dropout(emb_prem)
        emb_hypo = self.dropout(emb_hypo)
        
        enc_out, enc_state = self.encoder(emb_prem, prem_lengths)
        dec_out, dec_state, attention = self.decoder(emb_hypo, enc_out, enc_state, mask_prem, hypo_len=hypo_lengths)
        
        return dec_out, attention



# Utility functions

def count_parameters(model):
    """ Counts the number of learnable parameters in the model"""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def init_weights(model):
    """ Initializes weights of the model (except for pretrained embeddings) 
        with random values in a predefined interval. """
    for name, param in model.named_parameters():
        if name != "pretrained_embedding.weight":
            nn.init.uniform_(param, -0.1, 0.1)
        if "bias" in name:
            torch.nn.init.constant_(param, 0.0)
          
  
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    

def train(model, vocab,
          train_data,
          optimizer,
          criterion, 
          device,
          train_history=None, valid_history=None):
    
    """
    A function to perform a training loop on the data.
    
    Args:
        model: Seq2Seq model
        vocab: Vocab object
        training_data: a set of training examples
        optimizer: torch optimizer 
        criterion: loss function
        device: torch device
        train_history (list): a list of training loss
        valid_history (list): a list of validation loss
        
    Returns:
        (float) Average epoch loss
    
    """    

    
    model.train()
    iteration, num_batches = 0, 0
    temp_loss, epoch_loss = 0, 0
    history = []

    for batch_dictionary in generate_batches(train_data, vocab, device = device):  
        torch.cuda.empty_cache()
        iteration += 1
        num_batches += 1
        # reset gradients
        model.zero_grad()
                
        prem, hypo, len_prem, len_hypo = batch_dictionary.values()
        # compute output 
        log_probs, _ = model(prem, hypo, len_prem, len_hypo)
        num_classes = log_probs.size(-1) 
        
        # compute loss for a batch
        batch_loss = criterion(log_probs.view(-1, num_classes), 
                               hypo[:,1:].contiguous().view(-1))
        temp_loss += batch_loss.item()
        epoch_loss += batch_loss.item()
        batch_loss.backward()
        
        # clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        
        # update parameters
        optimizer.step()
        
        # logging and reporting
        history.append(temp_loss / iteration)
        
        if num_batches % 100 == 0:
            
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
            clear_output(True)
            print( temp_loss / iteration )
            temp_loss, iteration = 0, 0
            ax[0].plot(history, label='train loss')
            ax[0].set_xlabel('Batch')
            ax[0].set_title('Train loss')
            if train_history is not None:
                ax[1].plot(train_history, label='training history')
                ax[1].set_xlabel('Epoch')
            if valid_history is not None:
                ax[1].plot(valid_history, label='validation history')
            plt.legend()
            plt.show()
            
    return epoch_loss / num_batches   



def evaluate(model, vocab, dev_data, criterion, device):

    """
    A function to perform a validation loop on the data.
    
    Args:
        model: Seq2Seq model
        vocab: Vocab object
        dev_data: a set of validation examples
        criterion: loss function
        device: torch device
        
    Returns:
        (float) Average epoch loss
    
    """
    
    model.eval()
    epoch_loss, num_batches = 0, 0
    
    with torch.no_grad():
    
        for batch_dictionary in tqdm(generate_batches(dev_data, vocab, device = device)):  
        
            num_batches += 1
            
            # reset gradients
            model.zero_grad()
                    
            # compute probability distribution over vocabulary
            prem, hypo, len_prem, len_hypo = batch_dictionary.values()
            log_probs, _ = model(prem, hypo, len_prem, len_hypo)
            num_classes = log_probs.size(-1) 
            
            # compute loss for a batch
            batch_loss = criterion(log_probs.view(-1, num_classes), 
                               hypo[:,1:].contiguous().view(-1))
            epoch_loss += batch_loss.item()            
    
        return epoch_loss / num_batches
    
    
    
def predict(premise, model, voc, device, max_len=20):
    
    """ 
    Implements greedy decoding to generate the entailed sentence for a given premise.
    
    Args:
      premise: (str) premise
      model: seq2seq pretrained model
      voc: Vocabulary object
      device: precessing device
      max_len: (int) maximum hypothesis lengths 
    
    Returns:
      entailed_sentence: str
      attention_weights: array with size (hypothesis_len, premise_len)
      """
    
    input_word = torch.tensor([voc.vocabulary["<sos>"]]).view(1, -1).to(device)
    prem = voc.sentence2tensor(premise).view(1, -1).to(device)
    len_prem = torch.tensor([prem.size(1)]).to(device)
    
    model.eval()
    
    with torch.no_grad():
        
        # compute the mask and embed premise
        mask_prem = torch.ne(prem, .0)
        emb_prem = model.embed_sentence(prem)
        
        # retrieve premise encodings and the final network state
        enc_out, enc_state = model.encoder(emb_prem, len_prem)
        
        # generate the hypothesis and compute attention weights
        entailed_sentence, attns = [], []
        dec_state = enc_state
        hypo_len = torch.tensor([1]).to(device)
        for step in range(max_len):
            
            # embed the word of the entailed sentence
            dec_input = model.embed_sentence(input_word)
                
            # compute attention and the prediction
            dec_out, dec_state, attn =\
            model.decoder(dec_input, enc_out, dec_state, mask_prem, hypo_len=hypo_len)
            attns.append(attn.squeeze())
            
            # greedy decoding 
            _, topi = dec_out.squeeze().data.topk(1)
            topi = topi.view(-1)
            decoded_word = voc.vocabulary.lookup_tokens([int(topi)])[0]
            if decoded_word == "<eos>":
                break
    
            input_word = topi.detach().view(-1, 1)
            entailed_sentence.append(decoded_word)
            
        attns = torch.stack(attns).squeeze().cpu().numpy()
        
    return " ".join(entailed_sentence), attns
