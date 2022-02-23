import streamlit as st 
import pandas as pd 
from sklearn.metrics import cohen_kappa_score



def kappa_button():
    col1 = st.text_input('Coder 1',value='a a b')
    col2 = st.text_input('Coder 2', value='a a b')
    st.text('Kappa Score:') 
 
    try:
        return st.text(cohen_kappa_score(col1.split(),col2.split()))
    except ValueError:
        return st.text('Data must be the same length')
        
    


st.title('Cohen Kappa Calculator')
st.subheader("Calculate the inter-rater agreement between two coders using sklearn's `cohen_kappa_score` module")

st.subheader('How to use the calculator:')
st.text("""1. Copy the codes of Coder 1 into the Coder 1 text entry field and hit "Enter." 
2. Copy the codes for Coder 2 into the Coder 2 text entry field and hit "Enter."

🗒️ Make sure that the coding decisions between Coder 1 and Coder 2 are the same length.
""")

kappa_button()

st.write("""
## References
Boettger, R. K., & Palmer, L. A. (2010). Quantitative content analysis: Its use in technical communication. IEEE transactions on professional communication, 53(4), 346-357.

Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and psychological measurement, 20(1), 37-46.

Geisler, C., & Swarts, J. (2019). Coding streams of language: Techniques for the systematic coding of text, talk, and other verbal data. Ft. Collins, CO: WAC Clearinghouse.

Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E. (2011). Scikit-Learn: Machine Learning in Python. the Journal of machine Learning research, 12, 2825-2830.

scikit-learn developers. (2021). sklearn.metrics.cohen_kappa_score. Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html.
""")

st.write('© Ryan Omizo 2022')
