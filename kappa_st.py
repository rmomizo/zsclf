import streamlit as st 
import pandas as pd 
from sklearn.metrics import cohen_kappa_score
from scipy.stats import chi2, chi2_contingency
       

def chi():
    """
    Python code adapted from Brownlee (June 15, 2018)
    """
    st.text('This chi2 calculator assumes that your data is formatted as a contingency table:')
    st.markdown("""
    |sample|value1|value2|value3|
    |------|------|------|------|
    |s1|10|20|30|
    |s2|10|15|25|
    """)
    
    st.write("""
    1. Input the significant value (default/max value is .05)
    2. Copy the codes of your first sample into the Sampe 1 text entry field and hit "Enter." 
    3. Copy the codes for your second sample into the Sample 2 text entry field and hit "Enter."
    🗒️ Make sure that the coding decisions between Sample 1 and Sample 2 are numerical values. 
       """)
    significance = st.number_input('Input significance value (default/max value is .05)', value=.05,max_value=.05,min_value=None, step=.001)
    col1 = st.text_input('Sample 1',value='10 20 30')
    col2 = st.text_input('Sample 2', value='10 15 25')

    s1 = [int(c) for c in col1.split()]
    s2 = [int(c) for c in col2.split()]
       
    chi, p_val, dof, ex = chi2_contingency([s1,s2], correction=False)
    p = 1 - significance
    crit_val = chi2.ppf(p, dof)

    st.write('p-value: ', p_val)
    st.write('degree of freedom: ',dof) 
    st.write('\n chi2 test statistic: ',chi, 'critical value: ',crit_val,)
    st.write("For an extended discussion of using chi2 tests for homogeneity for qualitative coding, see [Geisler and Swarts (2019)](https://wac.colostate.edu/docs/books/codingstreams/chapter9.pdf)")

def kappa():
    st.text("""
    1. Copy the codes of Coder 1 into the Coder 1 text entry field and hit "Enter." 
    2. Copy the codes for Coder 2 into the Coder 2 text entry field and hit "Enter."
    🗒️ Make sure that the coding decisions between Coder 1 and Coder 2 are the same length.
       """)
    
    col1 = st.text_input('Coder 1',value='a a b')
    col2 = st.text_input('Coder 2', value='a a b')
 
    try:
        st.text('Kappa Score:') 
        st.text(cohen_kappa_score(col1.split(),col2.split()))
        st.markdown("For more an extended presentation on Cohen's Kappa see Hart-Davidson (2014), [Using Cohen's Kappa to Gauge Interrater Reliability](https://www.slideshare.net/billhd/kappa870)")
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

st.subheader('How to use the calculator')


main()

st.write("""
## References/Code Consulted
Boettger, R. K., & Palmer, L. A. (2010). Quantitative content analysis: Its use in technical communication. IEEE transactions on professional communication, 53(4), 346-357.

Brownlee, J. (June 15 2018). A Gentle Introduction to the Chi-Squared Test for Machine Learning. Retrieved from https://machinelearningmastery.com/chi-squared-test-for-machine-learning/.

Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and psychological measurement, 20(1), 37-46.

Geisler, C., & Swarts, J. (2019). Coding streams of language: Techniques for the systematic coding of text, talk, and other verbal data. Ft. Collins, CO: WAC Clearinghouse.

Hart-Davidson, William. (2014). “Using Cohen’s Kappa to Gauge Interrater Reliability.” Education, 10:44:25 UTC. https://www.slideshare.net/billhd/kappa870.

McHugh, Mary L. (2012 October 15). “Interrater Reliability: The Kappa Statistic.” Biochemia Medica 22, no. 3: 276–82.

Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E. (2011). Scikit-Learn: Machine Learning in Python. the Journal of machine Learning research, 12, 2825-2830.

scikit-learn developers. (2021). sklearn.metrics.cohen_kappa_score. Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html.

scipy Community. (2022). scipy.stats.chi2_contingency. Retrieved from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html.

Scott, William A. “Reliability of Content Analysis: The Case of Nominal Scale Coding.” The Public Opinion Quarterly 19, no. 3 (1955): 321–25.

Virtanen, P., Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt & SciPy 1.0 Contributors. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature methods, 17(3), 261-272
""")

st.write('© Ryan Omizo 2022')
