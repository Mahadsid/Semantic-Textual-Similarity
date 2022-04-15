import pandas as pd
import numpy as np
import tensorflow as tf       
import tensorflow_hub as hub 

model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(model_url)

df = pd.read_csv("Precily_Text_Similarity.csv")
df.head(5)
df.shape

def encod(input):
  return model(input)

from numpy import dot                                           
from numpy.linalg import norm 

result = []                                                       
for i in range(len(df)):
    texts = [df['text1'][i], df['text2'][i]]               
    texts_encod = encod(texts)                          
    ar_nd = tf.make_ndarray(tf.make_tensor_proto(texts_encod))
    cos_sim = dot(ar_nd[0], ar_nd[1])/(norm(ar_nd[0])*norm(ar_nd[1]))
    result.append(cos_sim) 

len(result)

sm_sc = pd.DataFrame(result, columns = ['Similarity_Score'])  
sm_sc.head(5)

df = df.join(sm_sc)
df.head()

df['Similarity_Score'] = df['Similarity_Score'] + 1    
df.head()

df['Similarity_Score'] = df['Similarity_Score']/df['Similarity_Score'].abs().max()
df.head()


