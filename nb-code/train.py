import math
import numpy as np
import json
import utils
from sklearn.model_selection import train_test_split


def count_ngrams_in_lang(freqs,langid):
    """
    Takes in:
    freqs: dictionary of ngram frequencies,
    langid: language id

    Returns:
    n : the number of entries with that langid in freqs
    """
    n = len([key for key in freqs.keys() if key[1]==langid])
    return n

def logpriors(freqs,langids):
    """
    Takes in:
    freqs: dictionary of ngram frequencies,
    langids: list of language ids

    returns:
    p = dictionary with entries (langid):logprior
    """
    prob ={}
    count = {}
    all_pairs = len(freqs)
    for id in langids:
        count[id]= count_ngrams_in_lang(freqs,id)
        prob[id] = math.log(count[id]/all_pairs)
    return prob,count

def loglikelihoods(freqs, langids):
    """
    Takes in:
    freqs: dictionary of ngram frequencies,
    langid: list of language ids

    Returns:
    l = dictionary with entries (ngram,langid):loglikelihood
    """
    _,p = logpriors(freqs,langids)
    l = {}
    for key in freqs.keys():
        _,id = key
        #print(p[id])
        likelihood=freqs[key]/p[id]
        l[key]=math.log(likelihood)
    return l

def train(corpus,labels,n,langids):
    """
    Takes in:
    corpus: the training corpus
    labels: the lables for the training corpus
    n: the n-gram length
    Returns:
    p: logprior dictionary
    l: loglikelihoods dictionary
    """
    freqs = utils.count_frequencies(corpus,labels,n)
    p,_ = logpriors(freqs,langids)
    l = loglikelihoods(freqs,langids) 
    return p,l


def save_to_json(corpus,labels,n,langids):
    """
    Takes in:
    corpus: the training corpus
    labels: the lables for the training corpus
    n: the n-gram length    

    Saves the logprior and loglikelihoods to json file
    """
    foldername="json_dicts/"
    p,l=train(corpus,labels,n,langids)
    utils.dict_to_json(foldername+"priors.json",p)
    utils.dict_to_json(foldername+"likelihoods.json",l)

LANGUAGES = ["eng","fr","pl"]

eng_corpus = utils.file2sentences('corpi/english.txt')
fr_corpus = utils.file2sentences('corpi/french.txt')
pl_corpus = utils.file2sentences('corpi/polish.txt')

corpus = np.array(eng_corpus+fr_corpus+pl_corpus)
labels = np.array(['eng']*len(eng_corpus)+['fr']*len(fr_corpus)+['pl']*len(pl_corpus))

corpus_train, corpus_test, label_train, label_test = train_test_split(corpus, labels, test_size=0.15, random_state=42)


save_to_json(corpus_train,label_train,2,LANGUAGES)