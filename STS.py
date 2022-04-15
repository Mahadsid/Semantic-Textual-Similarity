import pandas as pd
import numpy as np
import tensorflow as tf       
import tensorflow_hub as hub 
from numpy import dot                                           
from numpy.linalg import norm 
from scipy.special import expit


def encod(input):
    model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(model_url)
    return model(input)


def Semantic_textual_similarity(text1, text2):
  
    result = []                                                       
    
    texts = [text1, text2]               
    texts_encod = encod(texts)                          
    ar = tf.make_ndarray(tf.make_tensor_proto(texts_encod))
    cos_sim = dot(ar[0], ar[1])/(norm(ar[0])*norm(ar[1]))
    result.append(cos_sim)

    sig = expit(result)
    
    return sig

