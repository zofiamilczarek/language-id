import json
from ast import literal_eval

def preprocess(txt):
    """ 
        Takes in:
        txt: a string
        Returns:
        txt: processed string
    """
    txt = ''.join([i for i in txt if not i.isdigit()])
    txt = txt.replace("?",".")
    txt = txt.replace("!",".")
    txt = txt.replace("»","")
    txt = txt.replace("«","")
    txt = txt.replace(":","")
    txt = txt.replace(";","")
    txt = txt.replace("...",".")
    txt = txt.replace("…",".")
    txt = txt.replace("_",".")
    txt = txt.replace("\n"," ")
    txt = txt.replace("  "," ")
    txt = txt.replace("-","")
    txt = txt.replace("\"","")
    txt = txt.replace("/","")
    txt = txt.replace("„","")
    txt = txt.replace("*","")
    return txt

def file2sentences(filename):
    """ 
        Takes in:
        filename: a string with a file path of a text file.
        Returns:
        sentences: a list of strings in the order of the textfile
    """
    txt = ""
    with open(filename,"r",encoding="utf-8") as f:
        txt = f.read()
 
    txt = preprocess(txt)
    sentences = txt.split(".")
    for i in range(len(sentences)):
        sentences[i] = sentences[i].strip()
        
    sentences = [x for x in sentences if x != ""]
    return sentences

def n_gramify(text,n):
    """
        Takes in:
        text: a string,
        n: an integer n 
        
        Returns:
        n_grams: the list of n-grams in the string
    """
    n_grams = []
    
    for i in range(len(text)-n+1):
        n_gram = ''
        for j in range(n):
            n_gram += text[i+j] 
        n_grams.append(n_gram)
    return n_grams

def count_frequencies(corpus,labels,n=2):
    """Takes in:
        corpus: list of sentences,
        lables: list of sentence labels,
        n: n-gram length

        Returns:
        frequencies : a dictionary with entries (gram,lang):count
    """
    frequencies = {}

    for sentence, label in zip(corpus,labels):
        for gram in n_gramify(sentence,n):
            pair = (gram,label)
            if pair in frequencies:
                frequencies[pair] += 1
            else:
                frequencies[pair] = 1

    return frequencies


def eval_key(k):
    try:
        return literal_eval(k)
    except ValueError:
        return k

def dict_to_json(filename,my_dict):
    """ Creates/saves my_dict into a json file with filename"""
    json_dict=json.dumps({str(k): v for k, v in my_dict.items()})
    with open(filename, "w") as outfile:
        outfile.write(json_dict)

def load_from_json(filename):
    """ Loads a json file and returns a dictionary from it"""
    f = open(filename)
    txt=f.read()
    json_dict = json.loads(txt)
    f.close()
    #json_dict=json.loads(json_dict)
    d = {eval_key(k): v for k, v in json_dict.items()}
    return d

