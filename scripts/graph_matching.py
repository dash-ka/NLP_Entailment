import numpy as np
from collections import defaultdict
import spacy
from spacy.tokens import Token
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

def spacy_preorder(T):
    Q, L = [],[]
    root = [t for t in T if t.dep_=="ROOT"][0]
    Q.append(root)
    while len(Q)>0:
        v = Q.pop()
        if v.dep_=="ROOT":
            v._.depth = 0
        else: 
            v._.depth = v.head._.depth + 1
        L.append(v)
        for child in list(v.children)[::-1]:
            Q.append(child)
    return L

def spacy_top_down(T):
    Q, L = [], []
    root = [t for t in T if t.dep_=="ROOT"][0]
    Q.append(root)
    while len(Q)>0:
        v = Q.pop(0)
        L.append(v)
        for child in list(v.children):
            Q.append(child)
    return L

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


def generate_graph(nlp_sentence):
    
    """
    nlp_sentence : spacy parsed sentence
    
    """
    Token.set_extension("depth", default=None, force=True)
    Token.set_extension("folded", default=False, force=True)
    Token.set_extension("idx_list", default=[], force=True)
    Token.set_extension("word_list", default=[], force=True)
        
    nlp_traversal = spacy_postorder(nlp_sentence)
    
    G = nx.DiGraph()   
    
    for token in nlp_traversal:
        token._.word_list = [token.text]
        token._.idx_list = [token.idx]
          
    for token in nlp_traversal:
        
        if not token._.folded:
            
            attr_dictionary = {
                "lemma": token.lemma_,
                "text": token._.word_list,
                "idx_list": token._.idx_list,
                "dep_label": token.dep_,
                "pos": "VERB" if token.pos_ == "AUX" else token.pos_,                
                "parent": token.head.idx if token.head != token else "ROOT"
            }
            
            # skipping punctuations
            if token.dep_ in ["punct"]:
                continue
                
            # collapsing noun phrase components and verb particels/auxiliaries
            if token.dep_ in ["det", "compound", "aux", "auxpass", "prt"]:
                token.head._.word_list.append(token.text) 
                token.head._.idx_list.append(token.idx)
                children = [child.idx for child in token.children]
                if children and token.idx in G.nodes:
                    for child_idx in children:
                        G.nodes[child_idx]["parent"] = token.head.idx
                        G.remove_edge(token.idx, child_idx)
                        G.add_edge(token.head.idx, child_idx)
                    G.remove_node(token.idx)
                continue
               
            # direct preposition folding into a relation
            if token.head.dep_ == "prep":
                attr_dictionary["dep_label"] = token.head.lemma_
                attr_dictionary["parent"] = token.head.head.idx
                token.head._.folded = True
                
            # adding a node with its desciptive features
            G.add_node(token.idx, 
                       idx_list = attr_dictionary["idx_list"],
                       text = attr_dictionary["text"],
                       lemma = attr_dictionary["lemma"],
                       pos = attr_dictionary["pos"],
                       dep_label = attr_dictionary["dep_label"],
                       parent = attr_dictionary["parent"])
            
            # adding a directed edge to G
            parent = attr_dictionary["parent"]
            if G.nodes[token.idx]["dep_label"] != "ROOT":
                G.add_edge(parent, token.idx)
                
    # adjusting the order of words joined within a single node  
    for node in G.nodes:
        sorted_indices = np.argsort(np.array(G.nodes[node]["idx_list"]))
        G.nodes[node]["text"] = " ".join(list(np.array(G.nodes[node]["text"])[sorted_indices]))
            
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
        ax.margins(x = 0.18, y = 0.09)
        ax.set_axis_off()
    else:
        plt.axis("off")
        
        

def retrieve_attention(nlp_prem, nlp_hypo, attention_weights):
    
    idx_prem = [tok.idx for tok in nlp_prem] 
    idx_hypo = [tok.idx for tok in nlp_hypo]
    attention = defaultdict(lambda:defaultdict(float))
    for i, h in enumerate(idx_hypo):
        for j, p in enumerate(idx_prem):
            attention[h][p] = attention_weights[i, j] 
    return attention


def attention_score(h, t, H_tree, T_tree, attention):       
    individual_weights = np.array([attention[h][t_idx] for t_idx in T_tree.nodes[t]["idx_list"]])
    return  np.sum(np.square(individual_weights)) / np.sum(individual_weights)


def score(idx_h, idx_t, H_tree, T_tree):
    
    h = H_tree.nodes[idx_h]
    t = T_tree.nodes[idx_t]
    children_h = list(H_tree.successors(idx_h))
    children_t = list(T_tree.successors(idx_t))
    root = [n for n in T_tree if T_tree.nodes[n]["dep_label"] == "ROOT"][0]
    predecessors_dict = nx.predecessor(T_tree, root)
    parent = predecessors_dict[idx_t]
    direct_parent = T_tree.nodes[parent[0]]["M"] if parent else "ROOT"
    if parent:
        if predecessors_dict[parent[0]]:
            indirect_parent = T_tree.nodes[predecessors_dict[parent[0]][0]]["M"]
        else:
            indirect_parent = "ROOT"
    else:
        indirect_parent = "ROOT"
    
    common_features = (int(h["lemma"] == t["lemma"]) +
                       int(h["pos"] == t["pos"]) + 
                       int(h["lemma"] == t["lemma"])*int(h["pos"] == t["pos"])+
                       int(h["parent"] == direct_parent) * int(h["dep_label"]==t["dep_label"]) + 
                       int(h["parent"] in [direct_parent, indirect_parent]) * int(h["pos"]==t["pos"]) / 2 +
                       int(h["parent"] in [direct_parent, indirect_parent]) * int(h["lemma"] == t["lemma"]) / 2 + 
                       len(set(children_h).intersection({T_tree.nodes[child]["M"] for child in children_t}))*int(h["pos"] ==t["pos"])
                       ) / (5 + len(children_h))
    
    return  common_features 


class GraphMatch():
    
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
            
    def match(self):
        prev_mapping, self.mapping = [], []
        for h in self.H_graph_traversal:
            match_scores = np.array(
                [score(h, t, self.H_graph, self.T_graph) for t in self.T_graph_traversal]
            )
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
                self.mapping.append((self.H_graph.nodes[h]["lemma"], self.T_graph.nodes[match_idx]["lemma"]))
                self.T_graph.nodes[match_idx]["M"] = h
                
        self.scores_array = np.array(self.scores_array)
        array_scores = np.max(self.scores_array, axis = 1)
        self.mapping = np.array(self.mapping)
        self.mapping[array_scores==0] = ["NULL","NILL"]
        self.mapping = list(map(tuple, self.mapping))
                
    def attention_match(self, attention):
        self.attention_mapping, self.attention_scores_array = [], []
        for h in self.H_graph_traversal:
            match_scores = np.array(
                [attention_score(h, t, self.H_graph, self.T_graph, attention) for t in self.T_graph_traversal]
            )
            self.attention_scores_array.append(match_scores)
            match_idx = self.T_graph_traversal[np.argmax(match_scores)]
            self.attention_mapping.append((self.H_graph.nodes[h]["lemma"], self.T_graph.nodes[match_idx]["lemma"]))
            self.T_graph.nodes[match_idx]["Ma"] = h
        self.attention_scores_array = np.array(self.attention_scores_array)
        


def classify_entailement(attention_match, graph_match, T_graph, H_graph):
    size_prem = T_graph.number_of_nodes()
    size_hypo = H_graph.number_of_nodes()
    non_trivial_match = {(node, mapped_node) for (node, mapped_node) in graph_match if node != mapped_node}
    related_terms, residual_pairs, inference = set(), set(), set()
    if non_trivial_match:
        related_terms = non_trivial_match.intersection(attention_match)
        residual_pairs = non_trivial_match.difference(related_terms) 
        inference = {(node, mapped_node) for (node, mapped_node) in attention_match if node != mapped_node}\
        .union(residual_pairs).difference(related_terms)
        premise_coverage = (size_hypo - len(residual_pairs)) / size_prem
    else:
        premise_coverage = size_hypo / size_prem
        
    print()
    print("Semantically related terms:", related_terms)
    print("Possible Inference:", inference.union(residual_pairs))
    print("Estimate for syntactic subsumption:", round(1 - premise_coverage, 3))
    print("Estimate for lexical semantics:", round(len(related_terms)/size_hypo, 3))
    print("Estimate for high order inference", round(len(residual_pairs)/size_hypo, 3))