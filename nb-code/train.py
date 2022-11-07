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


def save_to_json(dictionary,filename):
    """
    Takes in:
    corpus: the training corpus
    labels: the lables for the training corpus
    n: the n-gram length    

    Saves the logprior and loglikelihoods to json file
    """
    foldername="nb-code/json_dicts/"
    utils.dict_to_json(foldername+filename+".json",dictionary)

def create_trained_data(langids):
    #eng_corpus = utils.file2sentences("corpi/eng.txt")
    #fr_corpus = utils.file2sentences('corpi/fr.txt')
    #pl_corpus = utils.file2sentences('corpi/pl.txt')

    #a list of where each entry is a lang corpus : like eng corpus
    language_corpi = {}
    corpus=[]
    labels=[]
    for lang in langids:
        filename = 'nb-code/corpi/'+lang+'.txt'
        language_corpi[lang] = utils.file2sentences(filename)
        corpus=corpus+language_corpi[lang]
        labels=labels+[lang]*len(language_corpi[lang])
    labels = np.array(labels)
    corpus = np.array(corpus)

    print(language_corpi.keys())

    corpus_train, _, label_train, _ = train_test_split(corpus, labels, test_size=0.12, random_state=42)
    
    p,l=train(corpus,labels,2,langids)
    save_to_json(p,"priors")
    save_to_json(l,"likelihoods")
    return p,l
    



LANGUAGES = ["eng","fr","pl"]
#create_trained_data(LANGUAGES)



