import spacy
import matplotlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from spacy.tokens import Token
from collections import defaultdict

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



def generate_graph(nlp_sentence, 
                   noun_phrase = ["det", "compound", "amod", "acomp", "poss"],
                   predicate = ["aux", "auxpass", "prt", "acomp"],
                   skip_tokens = ["punct", "cc"],
                   fold_preps = True):
    
    """
    Args:
        nlp_sentence (Doc object) : spacy parsed sentence
        noun_phrase (List[str]): syntactic dependencies to collapse with the head token (assuming a NOUN)
        predicate (List[str]): syntactic dependencies to collapse with the head predicate
        fold_preps (bool): if to fold prepositions into arcs, rather than keeping them as nodes
        skip_tokens (List[str]): syntactic dependencies to skip in the graph
        
        Note: find the list of possible dependency labels below.
        
    """

    Token.set_extension("depth", default=None, force=True)
    Token.set_extension("word_list", default=None, force=True)
    Token.set_extension("idx_list", default=None, force=True)
    Token.set_extension("folded", default=None, force=True)
    
    nlp_traversal = tree_postorder(nlp_sentence)
    
    G = nx.DiGraph()
    for token in nlp_traversal:
        token._.word_list = [token.text]
        token._.idx_list = [token.idx]
        
    for token in nlp_traversal:
        if token.dep_ in skip_tokens:
            continue
            
        if not token._.folded:
            attr_dictionary = {
                        "lemma": token.lemma_,
                        "text": token._.word_list,
                        "idx_list": token._.idx_list,
                        "dep_label": token.dep_,
                        "pos": "VERB" if token.pos_ == "AUX" else token.pos_,                
                        "parent": token.head.idx if token.head != token else "ROOT"
                    }
                
            # collapsing noun phrase and predicate components
            if (token.dep_ in noun_phrase) or ((token.dep_ in predicate) and (token.head.pos_ in ["VERB"])):
                token.head._.word_list.append(token.text)
                token.head._.idx_list.append(token.idx)
                children = [child.idx for child in token.children]
                if children and token.idx in G.nodes:
                    for child_idx in children:
                        # set the parent to grandparent node
                        G.nodes[child_idx]["parent"] = token.head.idx
                        G.remove_edge(token.idx, child_idx)
                        G.add_edge(token.head.idx, child_idx)
                    G.remove_node(token.idx)
                continue
                
            # preposition folding into a relational arc
            if fold_preps:
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
        G.nodes[node]["idx_list"] = sorted(G.nodes[node]["idx_list"])
        
    return G


# Dependency Labels (English)
# ClearNLP / Universal Dependencies
# https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md  

"""
    "acl": "clausal modifier of noun (adjectival clause)",
    "acomp": "adjectival complement",
    "advcl": "adverbial clause modifier",
    "advmod": "adverbial modifier",
    "agent": "agent",
    "amod": "adjectival modifier",
    "appos": "appositional modifier",
    "attr": "attribute",
    "aux": "auxiliary",
    "auxpass": "auxiliary (passive)",
    "case": "case marking",
    "cc": "coordinating conjunction",
    "ccomp": "clausal complement",
    "clf": "classifier",
    "complm": "complementizer",
    "compound": "compound",
    "conj": "conjunct",
    "cop": "copula",
    "csubj": "clausal subject",
    "csubjpass": "clausal subject (passive)",
    "dative": "dative",
    "dep": "unclassified dependent",
    "det": "determiner",
    "discourse": "discourse element",
    "dislocated": "dislocated elements",
    "dobj": "direct object",
    "expl": "expletive",
    "fixed": "fixed multiword expression",
    "flat": "flat multiword expression",
    "goeswith": "goes with",
    "hmod": "modifier in hyphenation",
    "hyph": "hyphen",
    "infmod": "infinitival modifier",
    "intj": "interjection",
    "iobj": "indirect object",
    "list": "list",
    "mark": "marker",
    "meta": "meta modifier",
    "neg": "negation modifier",
    "nmod": "modifier of nominal",
    "nn": "noun compound modifier",
    "npadvmod": "noun phrase as adverbial modifier",
    "nsubj": "nominal subject",
    "nsubjpass": "nominal subject (passive)",
    "nounmod": "modifier of nominal",
    "npmod": "noun phrase as adverbial modifier",
    "num": "number modifier",
    "number": "number compound modifier",
    "nummod": "numeric modifier",
    "oprd": "object predicate",
    "obj": "object",
    "obl": "oblique nominal",
    "orphan": "orphan",
    "parataxis": "parataxis",
    "partmod": "participal modifier",
    "pcomp": "complement of preposition",
    "pobj": "object of preposition",
    "poss": "possession modifier",
    "possessive": "possessive modifier",
    "preconj": "pre-correlative conjunction",
    "prep": "prepositional modifier",
    "prt": "particle",
    "punct": "punctuation",
    "quantmod": "modifier of quantifier",
    "rcmod": "relative clause modifier",
    "relcl": "relative clause modifier",
    "reparandum": "overridden disfluency",
    "root": "root",
    "ROOT": "root",
    "vocative": "vocative",
    "xcomp": "open clausal complement"   
"""



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
    