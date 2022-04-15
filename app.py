import streamlit as st
import pandas as pd
import numpy as np

from STS import Semantic_textual_similarity

st.title('Semantic_textual_similarity')
st.caption('By - Muhammad Mahad')

with st.form(key='my_form_1'):
    text_1 = st.text_input(label='text1')
    submit_button = st.form_submit_button(label='Submit')

with st.form(key='my_form_2'):
    text_2 = st.text_input(label='text2')
    submit_button = st.form_submit_button(label='Submit')

sm_sc = Semantic_textual_similarity(text1=text_1, text2=text_2)

st.write("similarity score")
st.text(sm_sc[0])
