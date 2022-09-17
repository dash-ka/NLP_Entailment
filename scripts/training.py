import random
import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from IPython.display import clear_output
from torch.utils.data import Dataset, DataLoader


def count_parameters(model):
    """ Counts the number of learnable parameters in the model"""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def init_weights(model):
    """ Initializes weights of the model (except for pretrained embeddings) 
        with random values in a predefined interval. """
    for name, param in model.named_parameters():
        if name != "embedding.weight":
            nn.init.uniform_(param, -0.08, 0.08)

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
    
    
def collate_batch(batch, vocab):
    """
    Encode each pair of sentences and convert into torch.tensors;
    record the corresponding lengths for both premise and hypothesis;
    pad premise tensors to the max premise length in a batch;
    pad hypothesis tensors to the max hypothesis length in a batch.

    Args:
    batch: a batch with premise-hypothesis pairs
    vocab: Vocabulary object
    
    Returns: a dictionary 
    padded_prem: tensor with size (batch_size, max_premise_len) 
    padded_hypo: tensor with size (batch_size, 1 + max_hypothesis_len)
    premise_lengths: tensor with size (batch_size)
    hypothesis_lengths: tensor with size (batch_size)
    
    """
    premise, hypothesis, prem_lengths, hypo_lengths = [],[],[],[]
    
    for prem, hypo in batch:
        
        encode_prem = vocab.sentence2tensor(prem) # indexed tensor + eos_token
        encode_hypo = vocab.sentence2tensor(hypo) # indexed tensor + eos_token
        premise.append(encode_prem)
        hypothesis.append(encode_hypo)
        
        prem_lengths.append(encode_prem.size(0))
        hypo_lengths.append(encode_hypo.size(0))
        
        
    # padding sequences to max_sequence_length
    padded_prem = nn.utils.rnn.pad_sequence(premise, batch_first = True)
    padded_hypo = nn.utils.rnn.pad_sequence(hypothesis, batch_first = True)
    
    # prepend <sos> token to hypothesis sequences
    sos_tensor =  torch.tensor([vocab.vocabulary["<sos>"]]).repeat(padded_hypo.size(0), 1)
    padded_hypo = torch.cat((sos_tensor, padded_hypo), dim = -1)
    
    return {"premise": padded_prem,
            "hypothesis": padded_hypo,
            "premise_lengths": torch.tensor(prem_lengths),
            "hypothesis_lengths": torch.tensor(hypo_lengths)
           }


def generate_batches(dataset, vocab,
                     batch_size=32,
                     collate_fn = collate_batch,
                     shuffle=True,
                     drop_last=True,
                     device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. 
    Transfers all tensors on the correct device location.
    """
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch_size, 
                            collate_fn = lambda batch:collate_batch(batch, vocab),
                            shuffle=shuffle,
                            drop_last=drop_last)
    
    for batch_dictionary in dataloader:
        out_batch_dictionary = {}
        for name, tensor in batch_dictionary.items():
            out_batch_dictionary[name] = batch_dictionary[name].to(device)
        yield out_batch_dictionary
        
        
def create_w2v_matrix(vocabulary, wv):
    """ Initializes an embedding matrix for vocabulary terms with word2vec pretrained vectors,
        or random vectors in case no pretrained embedding exists for a word.
    
    Args:
    vocabulary : a vocabulary 
    wv : pretrained word2vec vectors
    
    Returns:
    w2v_embeddings: word embedding matrix with shape (vocabulary_size, 300)
    oov_set: a set of out-of-vocabulary terms
    """
    w2v_embeddings = np.zeros((len(vocabulary), 300))
    oov_set = set()
    oov_size = 0

    for word, idx in sorted(vocabulary.get_stoi().items(), key=lambda x:x[1]):
        try:
            w2v_embeddings[idx] = wv[word]
            
        except KeyError:
            oov_set.add(word)
            oov_size += 1
            w2v_embeddings[idx] = np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=(300,))
        
    print("Number of words not found in pretrained w2v model:", oov_size)
    w2v_embeddings = torch.Tensor(w2v_embeddings)
    return w2v_embeddings, oov_set


    
def train(model,
          vocab,
          train_data,
          optimizer,
          criterion, 
          device,
          train_history=None, valid_history=None):
    
    model.train()
    iteration, epoch_loss = 0, 0
    history = []

    for batch_dictionary in tqdm(generate_batches(train_data, vocab, device = device)):  
        
        iteration += 1
        
        # reset gradients
        model.zero_grad()
                
        # compute output 
        log_probs = model(*batch_dictionary.values())
        num_classes = log_probs.size(-1) 
        
        # compute loss for a batch
        batch_loss = criterion(log_probs.view(-1, num_classes), 
                               batch_dictionary["hypothesis"][:,1:].contiguous().view(-1))
        epoch_loss += batch_loss.item()
        batch_loss.backward()
        
        # clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        
        # update parameters
        optimizer.step()
        
        # logging and reporting
        history.append(epoch_loss / iteration)
        
        if iteration % 100 == 0:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
            clear_output(True)
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
            
    return epoch_loss / iteration



def evaluate(model, vocab, dev_data, criterion, device):
    
    model.eval()
    epoch_loss, iteration = 0, 0
    
    with torch.no_grad():
    
        for batch_dictionary in tqdm(generate_batches(dev_data, vocab, device = device)):  
        
            iteration += 1
            
            # reset gradients
            model.zero_grad()
                    
            # compute probability distribution over vocabulary
            log_probs = model(*batch_dictionary.values())
            num_classes = log_probs.size(-1) 
            
            # compute loss for a batch
            batch_loss = criterion(log_probs.view(-1, num_classes), 
                                   batch_dictionary["hypothesis"][:, 1:].contiguous().view(-1))
            epoch_loss += batch_loss.item()            
    
        return epoch_loss / iteration