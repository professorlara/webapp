from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import spacy
# Define the functions
def wordcount(text):
    words = text.split()
    return len(words)

def lines(text):
    res = ""
    for i in text:
        if(i.isupper()):
            res+="*"+i
        else:
            res+=i
    m=res.split("*")
    m.remove('')
    numlines = len(m)
    return numlines

def type_token_ratio(text):
    words = text.split()
    unique_words = set(words)
    ttr = len(unique_words) / len(words) if words else 0
    return ttr

def ngrams(text, n):
    vectorizer = CountVectorizer(ngram_range=(n, n))
    analyzer = vectorizer.build_analyzer()
    n_grams = analyzer(text)
    return len(n_grams)

def unique_ngrams(text, n):
    vectorizer = CountVectorizer(ngram_range=(n, n))
    analyzer = vectorizer.build_analyzer()
    n_grams = analyzer(text)
    return len(set(n_grams))



def wordclass(text, category):
    
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    
    counts = {
        'adjective': 0,
        'noun': 0,
        'base_verb': 0,
        'total_verb': 0,
        'preposition': 0,
        'personal_pronoun': 0,
        'non3rdpersonsingularpresent_verb': 0,
        '3rdpersonsingularpresent_verb': 0,
        'past_participle_verb': 0,
        'coordinating_conjunctions': 0
    }
    
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN']:
            counts['noun'] += 1
        elif token.tag_ == 'IN':
            counts['preposition'] += 1
        elif token.tag_ == 'VB':
            counts['base_verb'] += 1
            counts['total_verb'] += 1
        elif token.tag_ in ['VBD', 'VBG']:
            counts['total_verb'] += 1
        elif token.tag_ == 'VBN':
            counts['total_verb'] += 1
            counts['past_participle_verb'] += 1
        elif token.tag_ == 'VBP':
            counts['non3rdpersonsingularpresent_verb'] += 1
            counts['total_verb'] += 1
        elif token.tag_ == 'VBZ':
            counts['3rdpersonsingularpresent_verb'] += 1
            counts['total_verb'] += 1
        elif token.pos_ in ['ADJ']:
            counts['adjective'] += 1
        elif token.tag_ == 'CC':
            counts['coordinating_conjunctions'] += 1
        elif token.tag_ == 'PRP':
            counts['personal_pronoun'] += 1
    
    content_density = (counts['total_verb'] + counts['noun'] + counts['adjective']) / wordcount(text)
    past_participle_verb_freq = counts['past_participle_verb'] / wordcount(text)
    coordinating_conjunctions_freq = counts['coordinating_conjunctions'] / wordcount(text)
    
    counts['content_density'] = content_density
    counts['past_participle_verb_freq'] = past_participle_verb_freq
    counts['coordinating_conjunctions_freq'] = coordinating_conjunctions_freq
    
    return counts[category]
