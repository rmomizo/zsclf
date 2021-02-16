#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 11:02:02 2021

@author: ryanomizo
"""

import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import re
from collections import defaultdict
import ktrain
from ktrain import text

def tuple_to_dataframe(decision):
    df = pd.DataFrame()
    for (x,y) in decision:
        df[x] = [y]        
    return df

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def load_model(): 
#    textcat = TextCategorizer(model.vocab)
#    textcat.from_disk('spacy_fscope_classifier_gensim_web3')    
    zsl = text.ZeroShotClassifier()
    return zsl

#@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def add_labels():
    labels = st.sidebar.text_input('Enter your labels separated by whitespace',value='describe evaluate suggest question')
    return labels.split()
    
st.title("Zero Shot Classification Prototype")
st.text("This prototype app utilizes transformers from Huggine Face and the ktrain machine learning library.")

#@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def main():
    
    clf = load_model()
    labels = add_labels()
    nli = st.sidebar.text_input("Enter your query:", value='This sentence is')
    doc = st.sidebar.text_area("Enter or copy and paste sentences to classify; Separate multiple documents with <br>", value="Try out this app!")
    
    
    
    
    if re.findall('<br>',doc):
        docs = doc.split('<br>')
        predictions = clf.predict(docs, labels=labels, include_labels=True,
                nli_template=nli + "{}.",multilabel=False)
        dx = []
        for prediction in predictions:
            dx.append(tuple_to_dataframe(prediction))
            st.dataframe(pd.concat(dx))
    else:
        predictions = clf.predict(doc, labels=labels, include_labels=True,
                nli_template=nli + "{}.",multilabel=False)
        d = tuple_to_dataframe(predictions)
        st.dataframe(d)
    
        

    
    return predictions
    
if __name__ == '__main__':
	main()
    
    
    
    
    