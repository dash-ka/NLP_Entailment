import re, spacy, torch
import numpy as np
from string import punctuation
from collections import defaultdict
from graph_entity_embedding import generate_graph, draw_graph, score

from transformers import BartTokenizer, BartModel
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartModel.from_pretrained("facebook/bart-base")

nlp = spacy.load("en_core_web_lg")



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


def clean_text(text):
        text = re.sub("-", " ", text.lower())
        return " ".join([re.sub(r'(["+*/#%&\\_@*)(\[\]]+)', r"", word) for word in text.split()])
    

def word_id2tok_id(tokens):
        
        """ Map words ids to token ids produced with BartTokenizer.
        Args:
            tokens (List[str,]): a list of tokens output from pretrained tokenizer  
        Returns:
            map_tokens (defaultdict(List[int,])): a map from word index to token indices
        
        """
        map_tokens = defaultdict(list)
        map_tokens[0].append(0)
        word_index = 0
        for tok_idx, tok in enumerate(tokens[1:], start=1):
            if (tok.startswith("Ġ")) or (tok in punctuation):
                word_index += 1
            map_tokens[word_index].append(tok_idx)
        return map_tokens


def encode_with_transformer(seq, model, tokenizer, device):
    tokens = tokenizer.tokenize(seq)
    input_dict = tokenizer(seq, return_tensors="pt")
    hidden_states = model(**input_dict).encoder_last_hidden_state[0, 1:-1].detach()
    return hidden_states, tokens



def encode_with_seq2seq(seq, model, vocab, device):
    seq_enc = vocab.sentence2tensor(seq).view(1, -1).to(device)
    seq_len = torch.tensor([seq_enc.size(1)]).to(device)
    
    model.eval()
    
    with torch.no_grad():
            
        # compute the mask and embed premise
        mask_prem = torch.ne(seq_enc, .0)
        emb_prem = model.embed_sentence(seq_enc)
            
        # retrieve premise encodings and the final network state
        # encoding = [batch_size, seq_len, hidden_dim]
        encodings, _ = model.encoder(emb_prem, seq_len)
        
    return encodings[0]



def node_embeddings(G, encoder_hiddens, tokens=None):
        
        """ Use word2token map to retieve all token embeddings 
            associated with a given entity node.
            
        Args:
            G (networkx graph object): entity graph
            tokens (List[str,]): list of tokens output from pretrained BartTokenizer
            encoder_hiddens (torch.FloatTensor): encoded token representations
            
        Returns:
            dict: where key is the entity_idx and value is a list of token embeddings 
            
        """
        embeddings_dictionary = defaultdict(list)
        
        if tokens is not None:
            # associate word indices with token indices
            w2t_map = word_id2tok_id(tokens)
        
            for node in G:
                #if G.nodes[node]["pos"] in self.entity_pos:
                for node_idx in G.nodes[node]["idx_list"]:
                    for tok_idx in w2t_map[node_idx]:
                        embeddings_dictionary[node].append(encoder_hiddens[tok_idx])
                        
        else:
            for node in G:
                #if G.nodes[node]["pos"] in self.entity_pos:
                for node_idx in G.nodes[node]["idx_list"]:
                    embeddings_dictionary[node].append(encoder_hiddens[node_idx])
             
        return embeddings_dictionary
    
    
def build_graph_and_embed(sentence, encoding_fun, model, tokenizer, device):
    
    """
    Build dependency graph and compute averaged embeddings for nodes
    
    Args:
        sentence (str): a sentence
        model: an instance of encoder
        tokenizer: an instance of sentence tokenizer
        encoding_fun (function): a function that takes a sentence(str) in input
                                and outputs contextual token embeddings.
               
    Returns:
        G (networkx Graph object): dependency graph 
        nlp_sentence (spacy Doc): parsed sentence
        average_embedding (List[FloatTensor,]): a list of averaged contextual embeddings for graph nodes 
    
    """
    # 1) clean text and parse with Spacy
    sentence = clean_text(sentence)
    nlp_sentence = nlp(sentence)
    
    # 2) build a dependency graph from parsed sentence
    G = generate_graph(nlp_sentence)
    
    # 3) compute contextual embeddings 
    encoding = encoding_fun(sentence, model, tokenizer, device) 
    encoding, tokens = (encoding[0], encoding[-1]) if isinstance(encoding, tuple) else (encoding, None)
    
    # 4) aggregate token embeddings at a node level
    embeddings_dictionary = node_embeddings(G, encoding, tokens=tokens)
    
    # 5) compute averaged node embedding
    average_embedding = []
    for node, embeddings in embeddings_dictionary.items():
        if len(embeddings)>1:
            node_embedding = torch.mean(torch.stack(embeddings), dim=0)
            G.nodes[node]["embedding"] = node_embedding
            average_embedding.append(node_embedding)
            
        else:
            node_embedding = embeddings[0]
            G.nodes[node]["embedding"] = node_embedding
            average_embedding.append(node_embedding)
            
    return G, nlp_sentence, average_embedding



def retrieve_attention(nlp_prem, nlp_hypo, attention_weights):
    
    idx_prem = [tok._.idx for tok in nlp_prem] 
    idx_hypo = [tok._.idx for tok in nlp_hypo]
    attention = defaultdict(lambda:defaultdict(float))
    for i, h in enumerate(idx_hypo):
        for j, p in enumerate(idx_prem):
            attention[h][p] = attention_weights[i, j] 
    return attention


class GraphMatch:
    
    def __init__(self, premise_graph, hypothesis_graph):

        # dependency graphs
        self.T_graph = premise_graph
        self.H_graph = hypothesis_graph
        
        # derive tree traversal order
        self.H_graph_traversal = top_down(self.H_graph)
        self.T_graph_traversal = top_down(self.T_graph)
        
        # initialize the "match" attribute on nodes of the premise
        for idx_t in self.T_graph_traversal:
            self.T_graph.nodes[idx_t]["M"] = None
            
            
    def graph_match(self):
        
        prev_mapping, self.mapping = [], []
        for h in self.H_graph_traversal:
            match_scores = np.array(
                [score(h, t, self.H_graph, self.T_graph) for t in self.T_graph_traversal]
            )
            # get the node_id of the node in the premise graph with max score
            match_idx = self.T_graph_traversal[np.argmax(match_scores)]
            self.mapping.append((self.H_graph.nodes[h]["lemma"], self.T_graph.nodes[match_idx]["lemma"]))
            self.T_graph.nodes[match_idx]["M"] = h
                
        while prev_mapping != self.mapping :
            prev_mapping = self.mapping
            self.mapping, self.scores_array = [], []
            for h in self.H_graph_traversal:
                match_scores = np.array(
                    [score(h, t, self.H_graph, self.T_graph) for t in self.T_graph_traversal]
                )
                self.scores_array.append(match_scores)
                match_idx = self.T_graph_traversal[np.argmax(match_scores)]
                self.mapping.append((self.H_graph.nodes[h], self.T_graph.nodes[match_idx]))
                self.T_graph.nodes[match_idx]["M"] = h
                
        self.scores_array = np.max(np.array(self.scores_array), axis=1)
        self.mapping = np.array(self.mapping)
        self.mapping[self.scores_array==0] = ["NULL", "NILL"]
        self.mapping = list(map(tuple, self.mapping))
        self.scores_array =  np.around(self.scores_array, decimals=3)
        
                
    def attention_match(self, attention):
        """
        Args:
            attention (ndarray): a matrix with attention weights
                        
        """
        self.attention_mapping, self.attention_scores_array = [], []
        for h in self.H_graph_traversal:
            match_scores = np.array(
                [attention_score(h, t, self.H_graph, self.T_graph, attention) for t in self.T_graph_traversal]
            )
            self.attention_scores_array.append(match_scores)
            match_idx = self.T_graph_traversal[np.argmax(match_scores)]
            self.attention_mapping.append((self.H_graph.nodes[h], self.T_graph.nodes[match_idx]))
        self.attention_scores_array = np.max(np.array(self.attention_scores_array), axis=1)
        
        
    def embedding_match(self, emb_prem, emb_hypo):
        
        """
        Args:
            emb_prem (FloatTensor): averaged embedding for nodes in premise graph
            emb_hypo (FloatTensor): averaged embedding for nodes in hypothesis graph
        """
        self.embedding_mapping, self.embedding_scores = [], []
        prem_nodes, hypo_nodes = list(self.T_graph.nodes), list(self.H_graph.nodes)
        C = torch.zeros(len(hypo_nodes), (len(prem_nodes)))
        
        for i, src  in enumerate(prem_nodes):
            for j, trg in enumerate(hypo_nodes):
                sim = torch.cosine_similarity(emb_prem[i].view(1, -1),
                                              emb_hypo[j].view(1,-1))
                C[j, i] = sim
                
        topv, topi = torch.topk(C, dim=1, k=1)
        for trg_idx, (src_idx, sim_value) in enumerate(zip(topi, topv)):
            
            
            trg_node = self.H_graph.nodes[hypo_nodes[trg_idx]]
            src_node = self.T_graph.nodes[prem_nodes[src_idx]]
            self.embedding_mapping.append((trg_node, src_node))
            self.embedding_scores.append(np.round(sim_value.item(), 3))
            
        self.embedding_scores = np.array(self.embedding_scores)





def classify_entailment(graph_match, alternative_match, T_graph, H_graph):
    
    size_prem = T_graph.number_of_nodes()
    size_hypo = H_graph.number_of_nodes()
    
    non_trivial_match = set()
    for (node, mapped_node) in graph_match:
        if not isinstance(node, str):
            if node["lemma"] != mapped_node["lemma"]:
                non_trivial_match = non_trivial_match.union([(node["lemma"], mapped_node["lemma"])])
        else:
            non_trivial_match = non_trivial_match.union([(node, mapped_node)])
            
    related_terms, residual_pairs, inference = set(), set(), set()
    if non_trivial_match:
        alternative_pairs = {(node["lemma"], mapped_node["lemma"]) for (node, mapped_node) in alternative_match \
                             if node["lemma"] != mapped_node["lemma"]}
        related_terms = non_trivial_match.intersection(alternative_pairs)
        residual_pairs = non_trivial_match.difference(related_terms) 
        inference = {(node["lemma"], mapped_node["lemma"]) for (node, mapped_node) in alternative_match \
                     if node["lemma"] != mapped_node["lemma"]} \
        .difference(related_terms)#.union(residual_pairs)
        premise_coverage = (size_hypo - len(residual_pairs)) / size_prem
    else:
        premise_coverage = size_hypo / size_prem
        
    related_nodes = []
    for (node, image) in graph_match:
        if not isinstance(node, str):
            if (node["lemma"], image["lemma"]) in related_terms:
                related_nodes.append((node, image))
        
    print("Semantically related terms:", related_terms)
    print("Possible Inference:", inference)
    print("Estimate for syntactic subsumption:", round(1 - premise_coverage, 3))
    print("Estimate for lexical semantics:", round(len(related_terms)/size_hypo, 3))
    print("Estimate for high order inference", round(len(residual_pairs)/size_hypo, 3))


    return related_nodes
