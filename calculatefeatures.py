from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import spacy
import re
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



def wordclass(text):
    tags = []
    for word in text.split():
        if word.lower() in ['i', 'you', 'he', 'she', 'it', 'we', 'they']:
            tags.append(('PRP', 'personal_pronoun'))
        elif word.lower() in ['and', 'but', 'or', 'nor', 'for', 'so', 'yet']:
            tags.append(('CC', 'coordinating_conjunctions'))
        elif word.lower() in ['is', 'are', 'am', 'was', 'were']:
            tags.append(('VB', 'base_verb'))
        elif word.lower().endswith(('ed', 'ing')):
            tags.append(('VBD', 'past_participle_verb'))
        elif word.lower().endswith('s'):
            tags.append(('VBZ', '3rdpersonsingularpresent_verb'))
        else:
            tags.append(('NN', 'noun'))
    return tags

# Initialize counts dictionary
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

# Perform part-of-speech tagging
tags = pos_tag(text)

# Iterate through tags and update counts
for tag in tags:
    if tag[0] in ['NN', 'NNS', 'NNP', 'NNPS']:
        counts['noun'] += 1
    elif tag[0] == 'IN':
        counts['preposition'] += 1
    elif tag[0] == 'VB':
        counts['base_verb'] += 1
        counts['total_verb'] += 1
    elif tag[0] == 'VBD':
        counts['total_verb'] += 1
        counts['past_participle_verb'] += 1
    elif tag[0] == 'VBZ':
        counts['3rdpersonsingularpresent_verb'] += 1
        counts['total_verb'] += 1
    elif tag[1] == 'adjective':
        counts['adjective'] += 1
    elif tag[1] == 'coordinating_conjunctions':
        counts['coordinating_conjunctions'] += 1
    elif tag[1] == 'personal_pronoun':
        counts['personal_pronoun'] += 1

# Calculate total words in text
total_words = wordcount(text)

# Calculate additional metrics
content_density = (counts['total_verb'] + counts['noun'] + counts['adjective']) / total_words
past_participle_verb_freq = counts['past_participle_verb'] / total_words
coordinating_conjunctions_freq = counts['coordinating_conjunctions'] / total_words

# Add calculated metrics to counts dictionary
counts['content_density'] = content_density
counts['past_participle_verb_freq'] = past_participle_verb_freq
counts['coordinating_conjunctions_freq'] = coordinating_conjunctions_freq

# Return count of specified category
return counts[category]
