import streamlit as st 
import pandas as pd 
from sklearn.metrics import cohen_kappa_score
from scipy.stats import chi2, chi2_contingency
       

def chi():
       
    st.text("""1. Copy the codes of Sample 1 into the Coder 1 text entry field and hit "Enter." 
       2. Copy the codes for Sample 2 into the Sample 2 text entry field and hit "Enter."

       🗒️ Make sure that the coding decisions between Sample 1 and Sample 2 are integer values of the same length.
       """)

    col1 = st.text_input('Sample 1',value='10 20 30')
    col2 = st.text_input('Sample 2', value='10 15 25')

    s1 = [int(c) for c in col1.split()]
    s2 = [int(c) for c in col2.split()]
    chi, pval, dof, ex = chi2_contingency([s1,s2], correction=False)

    st.text('p-value is: ' + str(pval))
    significance = 0.05
    p = 1 - significance
    critical_value = chi2.ppf(p, dof)

    return st.write('Test statistic: ',chi, 'critical value: ',critical_value)

def kappa():
    st.text("""1. Copy the codes of Coder 1 into the Coder 1 text entry field and hit "Enter." 
       2. Copy the codes for Coder 2 into the Coder 2 text entry field and hit "Enter."

       🗒️ Make sure that the coding decisions between Coder 1 and Coder 2 are the same length.
       """)
    st.text('Kappa Score:') 
    col1 = st.text_input('Coder 1',value='a a b')
    col2 = st.text_input('Coder 2', value='a a b')
 
    try:
        st.text(cohen_kappa_score(col1.split(),col2.split()))
    except ValueError:
        st.markdown('<mark>Error: Data must be the same length</mark>', unsafe_allow_html=True)
        
    
def main():
    options = st.selectbox('What metric would you like to apply?',("Cohen's Kappa", 'chi2'))
    
 
    if options == "Cohen's Kappa":
        
        kappa()
    else:
        
        chi()

st.title("Cohen's Kappa and chi2 Calculator")
st.subheader("Calculate the inter-rater agreement between two coders using sklearn's `cohen_kappa_score` module or calculate the chi2 homogeneity of two samples with `scipy`")

st.subheader('How to use the calculator:')


main()

st.write("""
## References
Boettger, R. K., & Palmer, L. A. (2010). Quantitative content analysis: Its use in technical communication. IEEE transactions on professional communication, 53(4), 346-357.

Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and psychological measurement, 20(1), 37-46.

Geisler, C., & Swarts, J. (2019). Coding streams of language: Techniques for the systematic coding of text, talk, and other verbal data. Ft. Collins, CO: WAC Clearinghouse.

Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E. (2011). Scikit-Learn: Machine Learning in Python. the Journal of machine Learning research, 12, 2825-2830.

scikit-learn developers. (2021). sklearn.metrics.cohen_kappa_score. Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html.

scipy Community. (2022). scipy.stats.chi2_contingency. Retrieved from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html.

Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt & SciPy 1.0 Contributors. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature methods, 17(3), 261-272
""")

st.write('© Ryan Omizo 2022')
