import numpy as np
import random
import json
import re
from gensim.models import word2vec, KeyedVectors
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer

path = 'model_resources/GoogleNews-vectors-negative300.bin'
googlevecs = KeyedVectors.load_word2vec_format(path, binary=True) 

def prepare(comment):
    
    cleaned_comment = re.sub("[^a-zA-Z']", ' ', comment.lower())   
    return cleaned_comment        
    
  
def embed(comment, vec=googlevecs):
    
    wordnet_lemmatizer = WordNetLemmatizer()
    #splitcomment = comment.split()
    tk = TweetTokenizer()
    splitcomment = tk.tokenize(comment)
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

    """
    generate html with reply depending on 
    identified label(s) in prediction
    """

    toxines = {0: 'general toxicity', 1: 'severe toxicity', 
               2: 'obscenity', 3: 'threat(s)', 
               4: 'insult(s)', 5: 'identity hate'}          
      
    if not np.any(prediction):
        reply = "Your message has been sent!"        
    else:              
        detected = ", ".join([toxines[idx] for idx, p in enumerate(prediction) if p == True])
        reply = f"Your message could not be sent due to possible violations of our community guidelines. " \
                f"The following violations were detected: {detected}."           
    
    return reply 
    