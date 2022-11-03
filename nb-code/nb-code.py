
#https://towardsdatascience.com/an-efficient-language-detection-model-using-naive-bayes-85d02b51cfbd

import math
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *


LANGUAGES = ["eng","fr","pl"]


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

# log(P(lang))
def logpriors(freqs,langids=LANGUAGES):
    """
    Takes in:
    freqs: dictionary of ngram frequencies,
    langids: list of language ids

    returns:
    p = dictionary with entries (langid):probability
    """
    prob ={}
    count = {}
    all_pairs = len(freqs)
    for id in langids:
        count[id]=count_ngrams_in_lang(freqs,id)
        prob[id] = math.log(count[id]/all_pairs)
    return prob,count

#loglikelihood is log(P(gram|lang))
def loglikelihoods(freqs, langids=LANGUAGES):
    """
    Takes in:
    freqs: dictionary of ngram frequencies,
    langid: list of language ids

    Returns:
    l = dictionary with entries (ngram,langid):likelihood
    """
    _,p = logpriors(freqs,langids=LANGUAGES)
    l = {}
    for key in freqs.keys():
        _,id = key
        #print(p[id])
        likelihood=freqs[key]/p[id]
        l[key]=math.log(likelihood)
    return l


#log(P(L))=log(P(langid))+log(P(gram-1|langid))+...+log(P(gram-n|langid))
def probability(text,langid,priors,likelihoods):
    """
    Takes in:
    text: n_gramified text
    langid: the id code for the language
    Returns:
    prob: the logprobability of the text being in that language
    """
    #print(likelihoods)
    logprob = priors[langid]
    for gram in text:
        key = (gram,langid)
        if key in likelihoods.keys():
            logprob += likelihoods[key]
    return logprob

def get_key(dict,val):
    for key, value in dict.items():
        if val == value:
            return key
 
    return "key doesn't exist"


def train(corpus,labels,n,langids=LANGUAGES):
    """
    Takes in:
    corpus: the training corpus
    labels: the lables for the training corpus
    n: the n-gram length
    Returns:
    p: logprior dictionary
    l: loglikelihoods dictionary
    """
    freqs=count_frequencies(corpus,labels,n)
    p,_ = logpriors(freqs)
    l = loglikelihoods(freqs) 
    return p,l


def predict(text,priors,likelihoods,langids=LANGUAGES):
    """
    Takes in:
    text: text in the form of a sentence strings
    priors: the logprior dictionary
    likelihoods: the loglikelihood dictionary
    langids: the list of language ids
    Returns:
    lang: what language it is
    """
    grams = n_gramify(text,2)
    probabilities = {}
    for id in langids:
        probabilities[id]=probability(grams,id,priors,likelihoods)
    #print(probabilities)
    maxval = max(probabilities.values())
    return get_key(probabilities,maxval)



def test_alg(corpus,labels,langids=LANGUAGES):
    corpus_train, corpus_test, label_train, label_test = train_test_split(corpus, labels, test_size=0.30, random_state=42)
    priors,likelihoods = train(corpus_train,label_train,2)
    count_hits=0
    count_miss=0
    for sentence,label in zip(corpus_test,label_test):
        l_predict = predict(sentence,priors,likelihoods)
        if label == l_predict:
            count_hits+=1
        else:
            count_miss+=1
    return count_hits/len(corpus_test)




"""
DONE:

-> make dict with probabilities P(gram|fr):0.02 -> DONE
-> make dict with probabilities P(fr):0.25 -> DONE
-> make prediction func P(L):
    - gramifies the input text
    - uses the training data
    - if gram,lang pair not in training data - ignore
    - 
-> make training function
-> make algo more effective so that incorporates more ngrams


TODO:
->add more languages
->add bigger datasets
->divide into modules
    -make a function for prediction with and without training
    -always load from the json dicts
->unknown language - what to do to classify into the unknown class (threshhold)
->print confusion matrix



P(L)=P(langid|text) = P(langid)*P(gram-1|langid)*...*P(gram-n|langid)
log(P(L))=log(P(langid))+log(P(gram-1|langid))+...+log(P(gram-n|langid))

compute log(P(L)) for each language 
prediction is L from the highest log(P(L))

"""


eng_corpus = file2sentences('corpi/english.txt')
fr_corpus = file2sentences('corpi/french.txt')
pl_corpus = file2sentences('corpi/polish.txt')



corpus = np.array(eng_corpus+fr_corpus+pl_corpus)
labels = np.array(['eng']*len(eng_corpus)+['fr']*len(fr_corpus)+['pl']*len(pl_corpus))

corpus_train, corpus_test, label_train, label_test = train_test_split(corpus, labels, test_size=0.15, random_state=42)


priors,likelihoods = train(corpus_train,label_train,2)

msg1 = "Jestem Zofia i pochodzę z Warszawy. Lubię grać w piłkę i jeść zupę. Moja mama jest nauczycielka"
msg2 = "Every shade of light and dark, of truth, and of fiction which is the veil of truth, is allowable in a work of philosophical imagination. It is not all on the same plane; it easily passes from ideas to myths and fancies, from facts to figures of speech. It is not prose but poetry, at least a great part of it, and ought not to be judged by the rules of logic or the probabilities of history. The writer is not fashioning his ideas into an artistic whole; they take possession of him and are too much for him."
msg3 = "Bonjour, je suis Zoé et je veux vous présenter mon chat. Il s'appelle Poireau et a une très belle fourrure. J'adore le caresser toute la journée, c'est vraiment un petit gars génial."

msg = "Herzlichen Glueckwunsch zum Geburtstag :)"

#print(test_alg(corpus,labels))

print("The language is: "+str(predict(msg,priors,likelihoods)))




"""
freqs = count_frequencies(corpus_train,label_train)

n_fr =count_ngrams_in_lang(freqs,"fr")
n_pl = count_ngrams_in_lang(freqs,"pl")
n_en =count_ngrams_in_lang(freqs,"eng")

print("fr ngrams: "+str(n_fr))
print("pl ngrams: "+str(n_pl))
print("en ngrams: "+str(n_en))
print("sum ngrams: "+str(n_en+n_fr+n_pl))
print("all ngrams: "+str(len(freqs)))
print("P(langids) dict: "+str(logpriors(freqs)))
print(loglikelihoods(freqs))
"""
