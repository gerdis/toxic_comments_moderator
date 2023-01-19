import numpy as np
import random
import json
import re
from gensim.models import word2vec, KeyedVectors
import nltk
from nltk.stem import WordNetLemmatizer

path = 'model_resources/GoogleNews-vectors-negative300.bin'
googlevecs = KeyedVectors.load_word2vec_format(path, binary=True) 

def preprocess(comment):
    
    cleaned_comment = re.sub("[^a-zA-Z']", ' ', comment.lower())   
    return cleaned_comment        
    
  
def embed(comment, vec=googlevecs):
    
    wordnet_lemmatizer = WordNetLemmatizer()
    splitcomment = comment.split()
    commentlist = []
    
    verbs_lemmatized = [wordnet_lemmatizer.lemmatize(word, pos='v') for word in splitcomment]
    splitcomment = [wordnet_lemmatizer.lemmatize(word, pos='n') for word in verbs_lemmatized]
    
    for word in splitcomment:
        try:
            commentlist.append(vec[word])

        except KeyError:
            pass    
    
    return np.array(commentlist)

def single_prediction(embedding, model):
    
    model_input = embedding.reshape(1, embedding.shape[0], 300)
    prediction = np.round(model.predict(model_input, steps=1))
    return prediction
    
def give_feedback(prediction, comment):

    toxines = {0: 'general toxicity', 1: 'severe_toxicity', 
               2: 'obscenity', 3: 'threat(s)', 
               4: 'insult(s)', 5: 'identity_hate'}             
        
    html = ''

    if not np.any(prediction):
        reply = "Thank you for your message!"
        html = addContent(html, feedback(reply))
    else:                
        detected = [' '.join(toxines[idx].split('_')) for idx, p in enumerate(prediction) if p == True]
        found = ", ".join(detected)          
        reply = f"Your message could not be sent due to possible violations of our community guidelines. " \
                f"The following violations were detected: {found}"
        html = addContent(html, feedback(reply, violation=True))
    
    return f'<div>{html}</div>'
 

def feedback(text, violation=False):
    """Create HTML text box"""
    
    if violation:
        raw_html = '<div style="border-bottom:1px inset black;border-top:1px inset black;padding:8px;font-size: 28px;color:red;">' + str(
            text) + '</div>'    
    else:
        raw_html = '<div style="border-bottom:1px inset black;border-top:1px inset black;padding:8px;font-size: 28px;">' + str(
            text) + '</div>'
    return raw_html


def addContent(old_html, raw_html):
    """Add html content together"""

    old_html += raw_html
    return old_html 
    