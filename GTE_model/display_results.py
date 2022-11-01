import torch
import torch.nn as nn
import numpy as np
import re, spacy
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchtext.data.utils import get_tokenizer
from string import punctuation



def greedy_decoder(model, annotations, encoder_hidden, vocab, device):
    
    """ Generates a hypothesis for a given premise.
    Args:
    model: trained seq2seq model
    annotations: tensor with shape (1, premise_len, hidden_size) 
    encoder_hidden: tensor with shape (2, 1, hidden_size) 
    vocab: Vocabulary object
    device: device

    Returns:
    decoded_hypothesis: str
    attention_weights: array with size (hypothesis_len, premise_len)
    """
    
    # initiate decoder hidden_state with encoder final hidden state
    decoder_hidden = encoder_hidden

    # create a tensor for <sos> token and a tensor for token length
    token_len = torch.tensor([1]).to(device)
    input_word = torch.tensor([vocab.vocabulary["<sos>"]]).view(1, 1).to(device)
  
    decoded_hypothesis, attention_weigths = [], []
    
    # start to decode a sequence of at most 20 tokens
    for step in range(20):

        # embed a word of the target sequence
        decoder_input = model.embedding(input_word)
        
        # compute probability distribution, hidden state, and attention
        probabilities, decoder_hidden, att_weight = model.decoder(decoder_input, token_len, annotations, decoder_hidden)
        attention_weigths.append(att_weight.squeeze().cpu().numpy())
        
        # decode the word with max probability score, stop if it is <eos> token
        _, topi = probabilities.data.topk(1)
        topi = topi.view(-1)
        decoded_word = vocab.vocabulary.lookup_tokens([int(topi)])[0]
        if decoded_word == "<eos>":
            break
    
        input_word = topi.detach().view(-1, 1)
        decoded_hypothesis.append(decoded_word)

    return " ".join(decoded_hypothesis), np.array(attention_weigths) 


def predict(premise, model, vocab, device): 
     
    """ Implements greedy decoding to generate the entailed sentence for a given premise.
      Args:
      premise: str
      model: seq2seq model
      vocab: Vocabulary object
      device: precessing device
    
      Returns:
      entailed_sentence: str
      attention_weights: array with size (hypothesis_len, premise_len)
      """

    premise_tensor = vocab.sentence2tensor(premise).view(1, -1).to(device)
    premise_len = torch.tensor([premise_tensor.size(1)]).to(device)
    
    model.eval()
    
    with torch.no_grad():
        # embed premise
        premise_tensor = model.embedding(premise_tensor)
        
        # retrieve contextual representation of premise(annotations) and the final hidden state
        annotations, hidden_enc = model.encoder(premise_tensor, premise_len)
        
        # generate the hypothesis and compute attention weights
        entailed_sentence, att_weights = greedy_decoder(model, annotations, hidden_enc, vocab, device)
        
    return entailed_sentence, att_weights


def word_tokenizer(text):
    """ Minimal sentence cleaning and tokenization."""
    
    tokenizer = get_tokenizer("basic_english")
    text = re.sub("-", " ", text)
    tokens = [w if w in punctuation else re.sub("[" + punctuation + "]", "", w) for w in tokenizer(text)]
    return tokens



def display_attention(premise, hypothesis, attention):
    fig, ax = plt.subplots(figsize =(15,30))
    cax = ax.matshow(attention, cmap='bone')
    
    # Set up axes
    ax.tick_params(labelsize =16)
    ax.set_xticklabels([""] + word_tokenizer(premise) + ['<EOS>'], rotation=60)
    ax.set_yticklabels( [""] + hypothesis.split(" ")+['<EOS>'])

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()
    
    
def draw_alignment(premise, decoded_hypothesis, attention_weights, THRESHOLD_ATTENTION = 0.0, tag = None):
                   
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # DEFINE THE BIPARTITE GRAPH 
    
    def get_source_target_labels(PREM, HYPO, tag = tag):
         # PREM : str
        # HYPO : str
        
        nlp = spacy.load("en_core_web_sm")
        
       
        if tag == None:
            tokenized_prem = PREM.split() + ['<eos>']
            tokenized_hypo = HYPO.split() + ['<eos>']
            return tokenized_prem, tokenized_hypo
        
        elif tag == "pos_tag":
            pos_prem = [tok.pos_ for tok in nlp(PREM)] + ['<eos>']
            pos_hypo = [tok.pos_ for tok in nlp(HYPO)] + ['<eos>']
            return pos_prem, pos_hypo
        
        elif tag == "dep_tag":
            dep_prem = [tok.dep_ for tok in nlp(PREM)] + ['<eos>']
            dep_hypo = [tok.dep_ for tok in nlp(HYPO)] + ['<eos>']
            return dep_prem, dep_hypo
    
    src_labels, trg_labels = get_source_target_labels(premise, decoded_hypothesis)
    
    G = nx.DiGraph()
    
    # specify indices of nodes for tokens in premise and hypothesis
    nodes_source = range(1, len(src_labels) + 1)
    nodes_target = range(len(src_labels) + 1, len(src_labels + trg_labels) + 1)
    
    # add nodes to the graph
    G.add_nodes_from(nodes_source, bipartite = "src")
    G.add_nodes_from(nodes_target, bipartite = "trg") 
    
    # add word attribute to source and target nodes
    nx.set_node_attributes(G, dict(zip(G.nodes, src_labels)), "word")
    for i, node in enumerate(nodes_target):
        G.nodes[node]["word"] = trg_labels[i]   
    
    # retrieve words to set labels for nodes in the graph
    # labels : dict {1: 'this', 2: 'church', 3: 'choir', 4: 'sings',... }
    labels = nx.get_node_attributes(G, 'word')
    
    # add edges with weights represented by attention score
    edge_w = []
    for i, trg in enumerate(nodes_target):
        for j, src in enumerate(nodes_source):
            if attention_weights[i,j] > THRESHOLD_ATTENTION:
                edge_w.append(attention_weights[i,j])
                G.add_edge(trg, src, weight=attention_weights[i,j])
    
    # specify a bipartite layout and correct misalignment in target sentence      
    pos = nx.bipartite_layout(G, nodes_target, align = "horizontal", scale = 10, aspect_ratio = 5/2)
    new_pos_list = []
    for node_idx in nodes_target:
        new_pos_list.append(pos[node_idx])
    new_pos_list = sorted(new_pos_list, key= lambda x:x[0])
    for node_idx, coord in zip(nodes_target, new_pos_list):
        pos[node_idx] = coord
     
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # DRAW THE NETWORK
    
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # Inflate attention weights in terms of arrow width
    edge_wts = [20.0 * edge for edge in edge_w]
    
    # Choose the colormap
    # plt.colormaps() check out for colormaps
    cm = plt.cm.ocean_r
    
    # Draw round-corner boxes around words
    # fc = facecolor of the box, ec = edgecolor of the box 
    bb = dict(boxstyle="round,pad=0.3", fc="w", ec="green", alpha=0.8, mutation_scale=10)
    
    # Visualize graph components
    nx.draw_networkx_edges(G, pos, alpha=0.9, width=edge_wts, edge_color=edge_wts,
                           edge_cmap=cm, arrowstyle="wedge", arrowsize=30, connectionstyle='arc3')
    nx.draw_networkx_nodes(G, pos, node_size= 0.5, node_color="#210070", alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=18, bbox=bb, labels = labels)
    
    # Normalize color values (default is in 0-1)
    norm = matplotlib.colors.Normalize(vmin=THRESHOLD_ATTENTION, vmax=1)
        
    # Create ScalarMappable for colorbar
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    plt.colorbar(sm, shrink= 0.9)
    
    # Resize figure for label readibility
    ax.margins(x = 0.01, y = 0.09)
    fig.tight_layout()
    plt.axis("off")
    plt.show()
