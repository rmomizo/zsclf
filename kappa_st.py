import streamlit as st 
import pandas as pd 
from sklearn.metrics import cohen_kappa_score



def kappa_button():
    col1 = st.text_input('Coder 1',value='a a b')
    col2 = st.text_input('Coder 2', value='a a b')
    st.text('Kappa Score:') 
 
    return st.text(cohen_kappa_score(col1.split(),col2.split()))
    


st.title('Cohen Kappa Calculator')
st.subheader("Calculate the inter-rater agreement between two coders using sklearn's `cohen_kappa_score` module")

st.subheader('How to use the calculator:')
st.text("""1. Copy the codes of Coder 1 into the Coder 1 text entry field and hit "Enter." 
2. Copy the codes for Coder 2 into the Coder 2 text entry field and hit "Enter."
""")

kappa_button()

st.write("""
## References

Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and psychological measurement, 20(1), 37-46.

Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E. (2011). Scikit-Learn: Machine Learning in Python. the Journal of machine Learning research, 12, 2825-2830.

scikit-learn developers. (2021). sklearn.metrics.cohen_kappa_score. Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html.
""")

st.write('© Ryan Omizo 2022')
