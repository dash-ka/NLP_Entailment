import re, torch
import pandas as pd
from string import punctuation
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class EntailmentDataset(Dataset):

    def __init__(self, data):
        self.df = data
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        pair = self.df.iloc[idx]
        return pair[0], pair[1]

    @classmethod
    def load_dataset(cls, path, split):
        """Load a specified partition of the SNLI dataset
        
        Args:
        path (str): folder where the snli datasets are located
        split (str): one of "train", "dev", or "test"
        
        Returns:
        an instance of Dataset
        """
        data = pd.read_table(f"{path}\snli_1.0_{split}.txt", delimiter ="\t", index_col = "gold_label")\
             .loc[["entailment"], ["sentence1", "sentence2"]].dropna()
    
        return cls(data)


def word_tokenizer(text):
    """ Minimal sentence cleaning and tokenization."""
    
    tokenizer = get_tokenizer("basic_english")
    text = re.sub("-", " ", text)
    tokens = [w if w in punctuation else re.sub("[" + punctuation + "]", "", w) for w in tokenizer(text)]
    return tokens

    
class Vocabulary:
    
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
    
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
    
    @classmethod
    def build_vocabulary(cls, data):
        """Extracts vocabulary from training data
        
        Args:
        data (str, str): sentence_pairs
        
        Returns:
        An instance of Vocabulary class
        """
        
        vocabulary = build_vocab_from_iterator(cls.yield_tokens(data),
                                               specials = ["<unk>", "<sos>", "<eos>"])
        vocabulary.insert_token("<pad>", 0)
        vocabulary.set_default_index(vocabulary["<unk>"])
        return cls(vocabulary) # return an instance of the class

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
