import numpy as np
import re, spacy
import networkx as nx
from collections import defaultdict
from string import punctuation
from spacy.tokens import Token
from spacy import displacy
import matplotlib.pyplot as plt

from transformers import BartTokenizer, BartModel
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartModel.from_pretrained("facebook/bart-base")

nlp = spacy.load("en_core_web_lg")




def spacy_postorder(T):
    S, L = [],[]
    root = [t for t in T if t.dep_=="ROOT"][0]
    S.append(root)
    while len(S)>0:
        v = S.pop()
        if v.dep_=="ROOT":
            v._.depth = 0
        else: 
            v._.depth = v.head._.depth + 1
        L.append(v)
        for child in list(v.children):
            S.append(child)
    return L[::-1]


def preorder(T):
    stack, visited = [], []
    num = 0
    root = [n for n in T.nodes if T.nodes[n]["dep_label"]=="ROOT"][0]
    stack.append(root)
    while len(stack)>0:
        num += 1
        v = stack.pop()
        T.nodes[v]["order"] = num
        visited.append(v)
        for child in list(T.successors(v))[::-1]:
            stack.append(child)
    return visited


def bottom_up(T):
    Q, R, L = [], [], []
    # record the number of children as the node attribute
    for n in T.nodes:
        T.nodes[n]["children"] = len(T[n])
        
    # initialize the R queue with the root node
    root = [n for n in T.nodes if T.nodes[n]["dep_label"]=="ROOT"][0]
    R.append(root)
    
    # use R as auxiliary queue to search for leaves
    # descend the tree top-down collecting leaves in Q
    while len(R) > 0:
        v = R.pop(0)
        for child in T[v]:
            # if the node is a leaf, enqueue in Q
            if T.nodes[child]["children"] == 0:
                Q.append(child)
            else:
                R.append(child) 
    num = 0
    while len(Q) > 0:
        v = Q.pop(0) # dequeue the front node
        L.append(v)
        num += 1
        T.nodes[v]["order"] = num
        if v != root: # if node is not ROOT
            parent = list(T.predecessors(v))[0]
            # decrease the number of children by 1
            T.nodes[parent]["children"] = T.nodes[parent]["children"]-1
            # append the parent once the list of its children has been exhausted
            if T.nodes[parent]["children"] == 0:
                Q.append(parent)   
    return L

def top_down(T):
    Q, L = [], []
    root = [n for n in T.nodes if T.nodes[n]["dep_label"]=="ROOT"][0]
    Q.append(root)
    while len(Q)>0:
        v = Q.pop(0)
        L.append(v)
        for child in T[v]:
            Q.append(child)
    return L



def generate_graph(nlp_sentence, 
                   noun_phrase = ["det", "compound", "amod", "acomp", "poss"],
                   predicate = ["aux", "auxpass", "prt", "acomp", "neg"],
                   skip_tokens = ["punct", "cc"],
                   fold_preps = True):
    
    """
    Args:
        nlp_sentence (Doc object) : spacy parsed sentence
        noun_phrase (List[str]): syntactic dependencies to collapse with the head token (assuming a NOUN)
        predicate (List[str]): syntactic dependencies to collapse with the head predicate
        fold_preps (bool): if to fold prepositions into arcs, rather than keeping them as nodes
        skip_tokens (List[str]): syntactic dependencies to skip in the graph
        
    """

    Token.set_extension("depth", default=None, force=True)
    Token.set_extension("word_list", default=None, force=True)
    Token.set_extension("idx_list", default=None, force=True)
    Token.set_extension("idx", default=None, force=True)
    Token.set_extension("children", default=[], force=True)
    Token.set_extension("folded", default=None, force=True)
    

    for i, token in enumerate(nlp_sentence):
        token._.idx = i
    
    # order nodes following postorder traversal of the syntax tree
    nlp_traversal = spacy_postorder(nlp_sentence)
    
    G = nx.DiGraph()
    for token in nlp_traversal:
        token._.word_list = [token.text]
        token._.idx_list = [token._.idx]
        token._.children = [tok._.idx for tok in token.children]
      
    for token in nlp_traversal:
        
        if token.dep_ in skip_tokens:
            token.head._.children = list(set(token.head._.children) - set([token._.idx]))
            continue
            
        if not token._.folded:
            attr_dictionary = {
                        "lemma": token.lemma_,
                        "text": token._.word_list,
                        "idx_list": token._.idx_list,
                        "dep_label": token.dep_,
                        "pos": "VERB" if token.pos_ == "AUX" else token.pos_,                
                        "parent": token.head._.idx if token.head != token else "ROOT"
                    }
                
            # collapsing noun phrase and predicate components
            if (token.dep_ in noun_phrase) or ((token.dep_ in predicate) and (token.head.pos_ in ["VERB"])):
                token.head._.word_list += token._.word_list
                token.head._.idx_list += token._.idx_list
                token.head._.children = list(set(token.head._.children) - set([token._.idx]))
                token.head._.children += token._.children
                
                # if token to be collapsed has been added to graph and has dependent nodes
                if (len(token._.children) > 0) and (token._.idx in G.nodes):
                    for child_idx in token._.children:
                        # set the parent to grandparent node
                        G.nodes[child_idx]["parent"] = token.head._.idx
                        G.remove_edge(token._.idx, child_idx)
                        G.add_edge(token.head._.idx, child_idx)
                    G.remove_node(token._.idx)
                continue
                
            # fold prepositions into arcs
            if fold_preps and (token.head.dep_ == "prep"):
                # 1) set the preposition as the arc label 
                attr_dictionary["dep_label"] = token.head.lemma_
                # 2) change the reference to the parent node
                attr_dictionary["parent"] = token.head.head._.idx
                # 3) remove the folded node_idx from the parent children
                token.head.head._.children = list(set(token.head.head._.children) - set([token.head._.idx]))
                # 4) add all collapsed nodes idx's to the children list of the parent node
                token.head.head._.children += [token._.idx]
                # 5) mark the preposition node as folded
                token.head._.folded = True
                
            # adding a node with its desciptive features
            G.add_node(token._.idx, 
                       idx_list = attr_dictionary["idx_list"],
                       text = attr_dictionary["text"],
                       lemma = attr_dictionary["lemma"],
                       pos = attr_dictionary["pos"],
                       dep_label = attr_dictionary["dep_label"],
                       parent = attr_dictionary["parent"],
                       children = token._.children)
                    
            # adding a directed edge to G
            parent = attr_dictionary["parent"]
            if G.nodes[token._.idx]["dep_label"] != "ROOT":
                G.add_edge(parent, token._.idx)
                
    # adjusting the order of words joined within a single node  
    for node in G.nodes:
        sorted_indices = np.argsort(np.array(G.nodes[node]["idx_list"]))
        G.nodes[node]["text"] = " ".join(list(np.array(G.nodes[node]["text"])[sorted_indices]))
        G.nodes[node]["idx_list"] = sorted(G.nodes[node]["idx_list"])
        
    return G





def draw_graph(G, ax=None):
    
    # Tree Layout
    for v in preorder(G):
        if G.nodes[v]["dep_label"] == "ROOT":
            G.nodes[v]["depth"] = 0
        else:
            parent = list(G.predecessors(v))[0]
            G.nodes[v]["depth"] = G.nodes[parent]["depth"] + 1
                
    for n in G.nodes:
        G.nodes[n]["breadth"] = 0
            
    for v in bottom_up(G):
        
        # if v is a leaf node
        if len(G[v]) == 0:
            G.nodes[v]["breadth"] = 1
        if G.nodes[v]["dep_label"]!="ROOT":
            parent = list(G.predecessors(v))[0]
            G.nodes[parent]["breadth"] = G.nodes[parent]["breadth"] + G.nodes[v]["breadth"]
            
    sibling = 0
    for v in top_down(G):
        G.nodes[v]["y"] = - G.nodes[v]["depth"]
        if G.nodes[v]["dep_label"] == "ROOT":
            G.nodes[v]["x"] = 0
        else:
            parent = list(G.predecessors(v))[0]
            brothers = list(child for child in G[parent])
            if v == brothers[0]:
                G.nodes[v]["x"] = G.nodes[parent]["x"] 
                sibling  = v
            
            else:
                G.nodes[v]["x"] = G.nodes[sibling]["x"] + G.nodes[sibling]["breadth"] 
                sibling = v
                
    for v in bottom_up(G):
        children = list(G[v])
        if len(children)>0:
            G.nodes[v]["x"] = (G.nodes[children[-1]]["x"] + G.nodes[children[0]]["x"])/2 
    
    labels = nx.get_node_attributes(G, 'text') 
    edge_labels = dict()
    for (src, trg) in G.edges:
        edge_labels[(src, trg)] = G.nodes[trg]["dep_label"]
    
    pos = dict()
    for i in G.nodes:
        pos[i]= np.array([G.nodes[i]["x"], G.nodes[i]["y"]])
        
    # styling and drawing
    bb = dict(boxstyle="round, pad=0.3", fc="w", ec="green", alpha=0.9, mutation_scale=10)
    
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5, edge_color='green', ax=ax, connectionstyle="arc3")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=15, bbox=bb, ax=ax, rotate=False)
    nx.draw_networkx_nodes(G, pos, node_size=3500, node_color = 'green', alpha=0.3, ax=ax) 
    nx.draw_networkx_labels(G, pos, labels = labels, font_color='k', alpha=0.9, font_size=18, font_family='sans-serif', ax=ax)
    
    # Resize figure for label readibility
    if ax is not None:
        ax.margins(x = 0.30, y = 0.10)
        ax.set_axis_off()
    else:
        plt.axis("off")




class EntityEmbedding():
    
    def __init__(self, tokenizer, model, entity_pos=["NOUN", "PROPN"]):
        """
        Class to retrieve contextual embeddings for entity nodes.
        
        Args:
            entity_pos List[str,]: list of POS tags representing entities in the graph
            tokenizer : pretrained BartTokenizer
            model : pretrained BartModel 
            
        """
        self.entity_pos = entity_pos
        self.tokenizer = tokenizer
        self.model = model
        
        
    def standardize_text(self, text):
        """ Standardize the raw text, removing noisy symbols."""
        return " ".join([re.sub(r"([-+*/#%&\\_@*)(\[\]])", r"", word) for word in text.split()])
    

    def check_spacy_alignment(self, tokens):
        """ Verify if the spacy parsed sentence aligns with the Bart tokenized sequence."""
        
        sequence = re.sub(r"([;:,.!?])", r" \1", re.sub("Ġ", " ", ("".join(tokens))))
        try:
            assert len(sequence.split()) == len(nlp(sequence))
        except AssertionError:
            print("Number of tokens in the sequence does not match with Spacy parsed sentence!")
            print("Sequence lenght:", len(sequence.split()), " Spacy-parsed length:",len(nlp(sequence)))
            
            
            
    def word_id2tok_id(self, tokens):
        """ Map words to token_ids output from BartTokenizer"""
        
        map_tokens = defaultdict(list)
        map_tokens[0].append(0)
        word_index = 0
        for tok_idx, tok in enumerate(tokens[1:], start=1):
            if (tok.startswith("Ġ")) or (tok in punctuation):
                word_index += 1
            map_tokens[word_index].append(tok_idx)
        return map_tokens
    
        
        
    def node_embeddings(self, G, tokens, encoder_hiddens):
        
        """ 
        Use word2token map to retrieve token embeddings 
            associated with a given entity node.
            
        Args:
            G (networkx graph object): entity graph
            tokens (list): list of token_idx output from pretrained BartTokenizer
            encoder_hiddens (torch.FloatTensor): encoder hidden states one for each token
            
        Returns:
            dict: where key is the entity_idx and value is a list of token embeddings 
            
        """
        
        w2t_map = self.word_id2tok_id(tokens)
        embeddings_dictionary = defaultdict(list)
        for node in G:
            if G.nodes[node]["pos"] in self.entity_pos:
                for node_idx in G.nodes[node]["idx_list"]:
                    for tok_idx in w2t_map[node_idx]:
                        embeddings_dictionary[node].append(encoder_hiddens[tok_idx])
        return embeddings_dictionary

    
    
    def get_entity_embeddings(self, sequence):
        
        """
        Generate the graph and retrieve embeddings for nodes corresponding to entities.
        
        Returns:
             G (networkx graph object): entity relation graph 
             entity_embeddings (dict): dictionary of entities with associated token embeddings
             average_node_embedding (FloatTensor):average of token embeddings representing the entity
        """
        
        sequence = self.standardize_text(sequence)
        tokens = self.tokenizer.tokenize(sequence)
        input_dict = self.tokenizer(sequence, return_tensors="pt")
        hidden_states = self.model(**input_dict).encoder_last_hidden_state[0, 1:-1].detach()
        
        nlp_sentence = nlp(sequence)
        check_spacy_alignment(tokens)
        G = generate_graph(nlp_sentence)
        
        entity_embeddings = self.node_embeddings(G, tokens, hidden_states)
        average_embedding = []
        for node, embeddings in entity_embeddings.items():
            if len(embeddings)>1:
                average_embedding.append(torch.mean(torch.stack(embeddings),dim=0))
            else:
                average_embedding.append(embeddings[0])
        return G, entity_embeddings, torch.stack(average_embedding)





# EXAMPLE
#
# sequence = "The National Institute of Microbiology in Israel was established in 1797."
# ee = EntityEmbedding(tokenizer, model)
# g, embeddings_dict, average_embedding = ee.get_entity_embeddings(sequence)
# fig = plt.figure(figsize=(10, 8))
# draw_graph(g)


# Dependency Labels (English)
# ClearNLP / Universal Dependencies
# https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md  
