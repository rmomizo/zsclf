import streamlit as st 
import random
import pandas as pd
from scipy.stats.contingency import expected_freq
import nltk
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from scipy.stats import chi2 as scipy_chi2, chi2_contingency, chisquare, zscore
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit as logistic
import warnings
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.model_selection import train_test_split
import pickle
import base64
import plotly.express as px
from scipy import stats
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MultiLabelBinarizer, OrdinalEncoder, normalize
from sklearn.datasets import load_wine, fetch_20newsgroups, fetch_california_housing

def flatten_list(somelist):
        if any(isinstance(el, list) for el in somelist) == False:
            return somelist
        flat_list = list(itertools.chain(*somelist))
        return flat_list
        
def chi_goodness():
    """
    
    """
    st.title('chi2 Goodness of Fit Test')
    st.markdown("""
The chi2 goodness of fit test determines how likely categorical coding decisions are a result of chance or a result of some other differentiating factor. The chi2 goodness of fit test does so by comparing observed coding frequencies with expected coding frequencies where expected coding frequencies is simply the mean of total coding decisions. In other words, the chi2 goodness of fit test assumes that expected frequencies are uniform across coding categories (all coding decisions are equally likely).

The chi2 goodness of fit test statistic is given by:""")

    st.latex(r'''
             \chi^2 = \Sigma_{i=1}^{n}\frac{(0_i - E_i)^2}{E_i}
             ''')
    st.markdown("""### When to Use the chi2 Goodness of Fit Test?
The chi2 goodness of fit test works on frequency counts of categorically coded data. 
Geisler and Swarts (2019) recommend that qualitative coding researchers run chi2 significant tests as initial measurements as a matter of course to reveal "surprises" (significant contrasts) within data. Because of its widespread usage, chi2 test have been referred to as "omnibus" tests (Sharp, 2015).

The chi2 test of goodness of fit does have sample size limitations. Frequencies less than 5 return poor results. Pearson (1900), the originator of the test, recommends frequency counts of at leat 13 (see also the Scipy Community (2022) [chisquare documentation page](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html#r81ecfb019d82-3)) 
                
### R2 the chi2 Goodness of Fit Test?
The chi2 goodness of fit test is a hypothesis test. We begin by posing the null hypothesis H<sub>0</sub> that the observed distribution is a result of chance coding.

Accepting (yes, the coding results are a product of chance) or rejecting (the observed distributions are not a product of chance) the null hypothesis is determined by whether or not the chi2 test statistic meets the critical value of significance. If the chi2 test statistic meets or exceeds the critical value, then we can reject the null and claim that coding decisions are a result of some underlying difference in the data capture by the coding scheme. If the chi2 test statistic is lower than the critical value, then we cannot reject the null hypothesis that coding decisions are a product of chance. Critical values can be found in a lookup table and are a function of the degree of freedom and the significance level of the test. Geisler and Swarts (2019) use this resource: [http://www.z-table.com/chi-square-table.html](http://www.z-table.com/chi-square-table.html).

To determine the critical value for a chi2 goodness of fit test, we need to set a significance value and compute a p-value from the chi2 test statistic. The significance value indicates the threshold at which an observed value will be considered the result of chance. For example, if we declare a significance value of 0.05, we are setting a standard by which our observations must have a p-value of 0.05 and lower to allow us to reject the null hypothesis. P-values higher than 0.05 indicate that we cannot reject the null hypothesis and decisions are largely the result of chance. Signicance values of .01 would require that we be 99% confident that observations are not occuring by chance in order to reject the null hypothesis.

`QC Utils` calculats the chi2 test statistic, critical values, and pvalues automatically with `scipy.`
 
If we can reject the null hypothesis (i.e., the frequency distributions of coding decisions differ from expectation), then we are still left with the task of determining which coding category is driving deviance from the expected frequency counts. Sharp (2015) suggest "residual analysis" as a method to determine which coding categories are the most responsible for the skewness. "Raw residuals" (Sharp, 2015) are the difference between observed and expected frequencies.

Geisler and Swarts (2019) direct us to individual chi2 number that sum to the chi2 test statistic. These values are given by:""",unsafe_allow_html=True)
    
    st.latex(r'''
             (O - E)^2/E\newline
             \textrm{O is the observed coding frequency}\newline
             \textrm{E is the expected value}
             ''')
    st.markdown("""Coding categories with large values are more responsible for the distribution.                
                
### How to use the chi2 Calculator?
This chi2 calculator assumes that your data consists of a single frequency distribution:
    
|sample|value1|value2|value3|
|------|------|------|------|
|sample|37|75|98|
    
    """)
    st.caption("In this case, you copy and paste row-wise values")
    st.markdown('Or')
    st.markdown("""
    |value|sample|
    |-----|--------|
    |value1|37|
    |value2|75|
    |value3|98|
    """)
    st.caption("In this case, you copy and paste column-wise values")
    
    st.write('To use the chi2 calculator:')
    st.write("""
    1. Input the significant value (default is .05)
    2. Copy the values of your sample and paste into the Sample text entry field and hit "Enter." 
    
    ❗By default, expected frequencies are equally likely. 
       """)
    significance = float(st.text_input('Input significance value (default is .05)', value='.05'))
    st.caption("Significance values are often set to 0.05 or 0.01")
   
    col1 = st.text_input('Sample',value='37 75 98')
    st.caption('These values are used in Geisler and Swarts (2019, p. 328)')
    st.caption("📝 This app does not retain user data.")
    s1 = [int(c) for c in col1.split()]
    E = np.mean(s1)
    
    chis = [(s - E)**2/E for s in s1]
    
    chi, p_val = chisquare(s1)
    p = 1 - significance
    
    dof = len(s1)-1
    crit_val = scipy_chi2.ppf(p, dof)
    st.subheader('Results')
    c1 = st.container()
    c2, c3, c4, c5 = st.columns(4)
    
    c1.metric('p-value', str(p_val))
    c2.metric('Dataset Length',str(len(s1)))
    c3.metric('degree of freedom',"{:.2f}".format(len(s1)-1))
    
    residuals = [s - E for s in s1]
    st.markdown("""#### Residuals""")
    r_df = pd.DataFrame(residuals, columns=['residual'])
    st.bar_chart(r_df)
    
    chart_df = pd.DataFrame(chis, columns=['(O-E)**2/E'])
    st.markdown("""#### chi2 values""")
    st.bar_chart(chart_df)
    c4.metric('\n chi2 test statistic',"{:.5f}".format(chi)) 
    c5.metric('critical value',"{:.5f}".format(crit_val))
    
    st.markdown("""### Sample Python Code""")
    with st.expander('Sample Python Code'):
        st.code("""from scipy.stats import chisquare
import numpy as np
                
#Calculate chi2 test statistic and p-values
sample = [37, 75, 98]

alpha = 0.05 #significance value
chi2, p_value = chisquare(sample)

p = 1 - alpha
degree_of_freedom = len(sample)-1
critical_value = scipy_chi2.ppf(p, degree_of_freedom)

expected_values = np.mean(sample)

chi2s = [(s - expected_values)**2/expected_values for s in sample]
                
                """)
        st.write("For an extended discussion of using chi2 goodness of fit tests for qualitative coding, see [Geisler and Swarts (2019)](https://wac.colostate.edu/docs/books/codingstreams/chapter9.pdf)")
       
def chi_goodness_file_upload():
    """
    
    """
    st.title('chi2 Goodness of Fit Test')
    st.markdown("""
The chi2 goodness of fit test determines how likely categorical coding decisions are a result of chance or a result of some other differentiating factor. The chi2 goodness of fit test does so by comparing observed coding frequencies with expected coding frequencies where expected coding frequencies is simply the mean of total coding decisions. In other words, the chi2 goodness of fit test assumes that expected frequencies are uniform across coding categories (all coding decisions are equally likely).

The chi2 goodness of fit test statistic is given by:""")

    st.latex(r'''
             \chi^2 = \Sigma_{i=1}^{n}\frac{(0_i - E_i)^2}{E_i}
             ''')
    st.markdown("""### When to Use the chi2 Goodness of Fit Test?
The chi2 goodness of fit test works on frequency counts of categorically coded data. 
Geisler and Swarts (2019) recommend that qualitative coding researchers run chi2 significant tests as initial measurements as a matter of course to reveal "surprises" (significant contrasts) within data. Because of its widespread usage, chi2 test have been referred to as "omnibus" tests (Sharp, 2015).

The chi2 test of goodness of fit does have sample size limitations. Frequencies less than 5 return poor results. Pearson (1900), the originator of the test, recommends frequency counts of at leat 13 (see also the Scipy Community (2022) [chisquare documentation page](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html#r81ecfb019d82-3)) 
                
### How to Interpret the chi2 Goodness of Fit Test?
The chi2 goodness of fit test is a hypothesis test. We begin by posing the null hypothesis H<sub>0</sub> that the observed distribution is a result of chance coding.

Accepting (yes, the coding results are a product of chance) or rejecting (the observed distributions are not a product of chance) the null hypothesis is determined by whether or not the chi2 test statistic meets the critical value of significance. If the chi2 test statistic meets or exceeds the critical value, then we can reject the null and claim that coding decisions are a result of some underlying difference in the data capture by the coding scheme. If the chi2 test statistic is lower than the critical value, then we cannot reject the null hypothesis that coding decisions are a product of chance. Critical values can be found in a lookup table and are a function of the degree of freedom and the significance level of the test. 

To determine the critical value for a chi2 goodness of fit test, we need to set a significance value and compute a p-value from the chi2 test statistic. The significance value indicates the threshold at which an observed value will be considered the result of chance. For example, if we declare a significance value of 0.05, we are setting a standard by which our observations must have a p-value of 0.05 and lower to allow us to reject the null hypothesis. P-values higher than 0.05 indicate that we cannot reject the null hypothesis and decisions are largely the result of chance. Signicance values of .01 would require that we be 99% confident that observations are not occuring by chance in order to reject the null hypothesis.

`QC Utils` calculats the chi2 test statistic, critical values, and pvalues automatically with `scipy.`
 
If we can reject the null hypothesis (i.e., the frequency distributions of coding decisions differ from expectation), then we are still left with the task of determining which coding category is driving deviance from the expected frequency counts. Sharp (2015) suggest "residual analysis" as a method to determine which coding categories are the most responsible for the skewness. "Raw residuals" (Sharp, 2015) are the difference between observed and expected frequencies.


Geisler and Swarts (2019) direct us to individual numbers that contribute to the chi2 test statistic. These values are given by:""",unsafe_allow_html=True)
    
    st.latex(r'''
             (O - E)^2/E\newline
             \textrm{O is the observed coding frequency}\newline
             \textrm{E is the expected value}
             ''')
    st.markdown("""### How to use the chi2 Goondess of Fit Calculator

1. Input the significant value (default is .05)
2. Upload your .csv, .xls, or .xlsx file. Insure that you name your column of values "sample" like the example below. 
    
    
|value|sample|
|-----|--------|
|value1|37|
|value2|75|
|value3|98|

❗By default, expected frequencies are equally likely across classes. 
       """)
    significance = float(st.text_input('Input significance value (default is .05)', value='.05'))
    st.caption("Significance values are often set to 0.005, 0.05, and 0.1")
   
    uploaded = st.file_uploader('Upload your .csv, .xls, or .xlsx file.')
    st.caption("📝 This app does not retain user data.")
    st.markdown("""### Sample Python Code""")
    with st.expander('Sample Python Code'):
        st.code("""from scipy.stats import chi2, chisquare, chi2_contingency
import numpy as np
                
#Calculate chi2 test statistic and p-values
sample = [37, 75, 98]

alpha = 0.05 #significance value
chi2, p_value = chisquare(sample)

degree_of_freedom = len(s1)-1
critical_value = scipy_chi2.ppf(p, degree_of_freedom)

expected_values = np.mean(sample)

chi2s = [(s - expected_values)**2/expected_values for s in sample]
                
                """)
        st.write("For an extended discussion of using chi2 goodness of fit tests for qualitative coding, see [Geisler and Swarts (2019)](https://wac.colostate.edu/docs/books/codingstreams/chapter9.pdf)")
    if uploaded != None:
        if uploaded.name.endswith('csv'):
            df = pd.read_csv(uploaded)
            s1 = df['sample']
                       
            chi, p_val = chisquare(s1)
            p = 1 - significance
            crit_val = scipy_chi2.ppf(p, len(s1)-1)
            st.subheader('Results')
            c1 = st.container()
            c2, c3, c4, c5 = st.columns(4)

            c1.metric('p-value', str(p_val))
            c2.metric('Dataset Length',str(len(s1)))
            c3.metric('degree of freedom',"{:.2f}".format(len(s1)-1)) 
            c4.metric('\n chi2 test statistic',"{:.5f}".format(chi)) 
            c5.metric('critical value',"{:.5f}".format(crit_val))
            
       
        elif uploaded.name.endswith('xls'):
            df = pd.read_excel(uploaded)
            s1 = df['sample']
                        
            chi, p_val = chisquare(s1)
            p = 1 - significance
            crit_val = scipy_chi2.ppf(p, len(s1)-1)
            st.subheader('Results')
            c1 = st.container()
            c2, c3, c4, c5 = st.columns(4)

            c1.metric('p-value', str(p_val))
            c2.metric('Dataset Length',str(len(s1)))
            c3.metric('degree of freedom',"{:.2f}".format(len(s1)-1)) 
            c4.metric('\n chi2 test statistic',"{:.5f}".format(chi)) 
            c5.metric('critical value',"{:.5f}".format(crit_val))
            
            
        else:
            st.write('Please upload a .csv or .xslx file')
            
def chi():
    """
    Python code adapted from Brownlee (June 15, 2018)
    """
    st.title('chi2 Test of Homogeneity')
    st.markdown("""Geisler and Swarts (2019) refer to the chi2 test of homogeneity as a test of association (see also Creswell, 2008). The chi2 test of homogeneity is the same as a the chi2 goodness of fit test. What differs is the assumptions of the hypothesis. The chi2 test of homogeneity determines if two distributions derive from the same population. 
                
For qualitative coding research, the chi2 test of homogeneity allows us to compare codes from independent samples. Let us say that we have a code book designed to identify the rhetorical moves of peer review. The codebook features 3 codes: Describe, Evaluate, and Suggest (for more on this coding research, see Omizo et al., 2021). 

We are applying the codebook to two different courses: a first year writing course and a first year biology course. Our data takes the form of a frequency table. Each row reflects a discipline (English or biology); each column contains the frequency count per code.

|sample|describe|evaluate|suggest|
|------|------|------|------|
|english|10|20|30|
|biology|10|15|25|""")


    st.caption(":memo: While the Omizo et al. (2021) codebook is real, the data used in this app is simulated.")
    st.markdown("""
The chi2 test of homogeneity helps us determine if the distributions of coding decisions are similar or different. 
                
### When to Use chi2 Test of Homogeneity?
The chi2 test of homogeneity should be used when comparing two samples with comparable codes. Samples should be independence (samples coded from different disciplines or genres; samples coded at different events).

### How to Interpret the chi2 Test of Homogeneity?
Like the chi2 goodness of fit test, we pose a null hypothesis that the two samples derive from the same population. If we can reject the null, then the two samples derive from different populations. If we cannot reject the null, then the two samples derive from the same population data. 

To prove or disprove the null hypothesis, we set a significance or p-value threshold (e.g., 0.05 or 0.01). If we set the significance threshold at 0.05, then the probability that observations are the result of chance should be 5% or lower. Inversely, we can claim that we are 95% confident that the observed values did not arise from random chance. 

Once we declare a significance value, we compute the chi square values of each coding category and total them to produce the chi2 test statistic. We derive the chi values with the following equation:""", unsafe_allow_html=True)

    st.latex(r'''
             \chi = (O - E)^2/E\newline
             \textrm{O is the observed coding frequency}\newline
             \textrm{E is the expected value}
             ''')

    st.markdown("""We calculate the expected value of each cell of the table by multiplying the row value with the column value and divide by the total number of coding decisions. For example, the marginal total first row of the coded peer review data is 60. The marginal total for the first column is 20. Thus:
                
    E = row1_total * col1_total / total_cases 
        
The expected values for the coded peer review data are:
    
|sample|Describe_exp|Evaluate_exp|Suggest_exp|
|------|------------|------------|-----------|
|English|10.9|19.09|30|
|biology|9.09|15.9|25|

The chi value of English-Descibe would be:
    
    (10 - 10.09)**2/10.09 = 0.00
    
The chi value of biology-Describe would be:
    
    (10-9.09)**2/9.09 = .091
    
The chi value of English-Evaluate is:
    
    (20 - 19.09)**2/19.09 = .043

The chi value for biology-Evaluate is:

    (15 -15.9)**2/15.9 = .050
    
The chi value of English-Suggest is:
    
    (30 - 30)**2/30 = 0.0
    
The chi square value of biology-Suggest is:
    
    (25 -25)**2/25 = 0.0
    
    
### How to Use the chi2 Test of Homogeneity?
                """)
    st.write('This chi2 calculator assumes that your data takes the form of a frequency table:')
    
    
    st.markdown("""
    |sample|value1|value2|value3|
    |------|------|------|------|
    |sample1|10|20|30|
    |sample2|10|15|25|
    """)
    st.caption("In this case, you copy and paste row-wise values")
    st.markdown('Or')
    st.markdown("""
    |value|sample 1|sample 2|
    |-----|--------|--------|
    |value1|10|10|
    |value2|20|15|
    |value3|30|25|
    """)
    st.caption("In this case, you copy and paste column-wise values")
       
    
       
    st.write('The chi2 calculator accepts the first row of your data in the Sample 1 field and the second row of your data in the Sample 2 field.')
    
    st.write('To use the chi2 calculator:')
    st.write("""
    1. Input the significant value (default value is .05)
    2. Copy the values of your first sample and paste into the Sample 1 text entry field and hit "Enter." 
    3. Copy the values for your second sample and paste into the Sample 2 text entry field and hit "Enter."
    ❗Samples 1 and Sample 2 must be numerical values. 
       """)
    significance = float(st.text_input('Input significance value (default value is .05)', value='.05'))
    col1 = st.text_input('Sample 1',value='10 20 30')
    col2 = st.text_input('Sample 2', value='10 15 25')
    st.caption("📝 This app does not retain user data.")

    s1 = [int(c) for c in col1.split()]
    s2 = [int(c) for c in col2.split()]
    
    df = pd.DataFrame({'sample1':s1,'sample2':s2})
    
    
    
    
    chi, p_val, dof, ex = chi2_contingency([df['sample1'],df['sample2']], correction=False)
    p = 1 - significance
    crit_val = scipy_chi2.ppf(p, dof)
    st.subheader('Results')
    
    cc1, cc2 = st.columns(2)
    cc1.markdown("""#### Observed Values""")
    cc1.dataframe(pd.DataFrame(df).transpose())
    
    
    cc2.markdown("""#### Expected Values""")
    ef = pd.DataFrame(ex,index=['sample1','sample2'])
    
    
    cc2.dataframe(ef)
    
    c1 = st.container()
    c2, c3, c4, c5 = st.columns(4)
    
    c1.metric('p-value', str(p_val))
    c2.metric('Dataset Length',str(len(df)))
    c3.metric('degree of freedom',"{:.2f}".format(dof)) 
    c4.metric('\n chi2 test statistic',"{:.5f}".format(chi)) 
    c5.metric('critical value',"{:.5f}".format(crit_val))
    
    st.markdown("""### Sample Python Code""")
    with st.expander('Sample Python Code'):
        st.code("""from scipy.stats import chi2_contingency

sample1 = [35,45,100]
sample2 = [45,40,200]
chi, p_val, dof, ex = chi2_contingency([sample1,sample2], correction=False)

alpha = 0.05
p = 1- alpha

critical_value = scipy_chi2.ppf(p, dof)
                """)
        st.write("For an extended discussion of using chi2 tests for homogeneity for qualitative coding, see [Geisler and Swarts (2019)](https://wac.colostate.edu/docs/books/codingstreams/chapter9.pdf)")

def chi_file_upload():
    """
    Python code adapted from Brownlee (June 15, 2018)
    """
    st.title('chi2 Test of Homogeneity')
    st.markdown("""Geisler and Swarts (2019) refer to the chi2 test of homogeneity as a test of association (see also Creswell, 2008). The chi2 test of homogeneity is the same as a the chi2 goodness of fit test. What differs is the assumptions of the hypothesis. The chi2 test of homogeneity determines if two distributions derive from the same population. 
                
For qualitative coding research, the chi2 test of homogeneity allows us to compare codes from independent samples. Let us say that we have a code book designed to identify the rhetorical moves of peer review. The codebook features 3 codes: Describe, Evaluate, and Suggest (for more on this coding research, see Omizo et al., 2021). 

We are applying the codebook to two different courses: a first year writing course and a first year biology course. Our data takes the form of a frequency table. Each row reflects a discipline (English or biology); each column contains the frequency count per code.

|sample|describe|evaluate|suggest|
|------|------|------|------|
|english|10|20|30|
|biology|10|15|25|""")

    st.markdown("""
The chi2 test of homogeneity helps us determine if distributions of coding decisions are similar or different.  
    
                
### When to Use chi2 Test of Homogeneity?
The chi2 test of homogeneity should be used when comparing two samples with comparable codes. Samples should be independence (samples coded from different disciplines or genres; samples coded at different events).

### How to Interpret the chi2 Test of Homogeneity?
Like the chi2 goodness of fit test, we pose a null hypothesis that the two samples derive from the same population. If we can reject the null, then the two samples derive from different populations. If we cannot reject the null, then the two samples derive from the same population data. 

To prove or disprove the null hypothesis, we set a significance or p-value threshold (e.g., 0.05 or 0.01). If we set the significance threshold at 0.05, then the probability that observations are the result of chance should be 5% or lower. Inversely, we can claim that we are 95% confident that the observed values did not arise from random chance. 

Once we declare a significance value, we compute the chi square values of each coding category and total them to produce the chi2 test statistic. We derive the chi values with the following equation:""",unsafe_allow_html=True)
    st.latex(r'''
             \chi = (O - E)^2/E\newline
             \textrm{O is the observed coding frequency}\newline
             \textrm{E is the expected value}
             ''')

    st.markdown("""We calculate the expected value of each cell of the table by multiplying the row value with the column value and divide by the total number of coding decisions. For example, the marginal total first row of the coded peer review data is 60. The marginal total for the first column is 20. Thus:
                
    E = row1_total * col1_total / total_cases 
        
The expected values for the coded peer review data are:
    
|sample|Describe_exp|Evaluate_exp|Suggest_exp|
|------|------------|------------|-----------|
|English|10.9|19.09|30|
|biology|9.09|15.9|25|

The chi value of English-Descibe would be:
    
    (10 - 10.09)**2/10.09 = 0.00
    
The chi value of biology-Describe would be:
    
    (10-9.09)**2/9.09 = .091
    
The chi value of English-Evaluate is:
    
    (20 - 19.09)**2/19.09 = .043

The chi value for biology-Evaluate is:

    (15 -15.9)**2/15.9 = .050
    
The chi value of English-Suggest is:
    
    (30 - 30)**2/30 = 0.0
    
The chi square value of biology-Suggest is:
    
    (25 -25)**2/25 = 0.0
    
    """)
    st.write('This chi2 calculator assumes that your data is in the form of a frequencey table:')
    
    
    st.markdown("""
    |values|sample 1|sample 2|
    |-------|------|------|
    |val1|30|30|
    |val2|20|30|
    |val3|40|15|
    |val4|24|20|
    
    """)
    
    st.write('To use the chi2 calculator:')
    st.write("""
    1. Input the significance value.
    2. Upload your frequency table as an .csv or .xlsx file. Make sure that the column names for your two samples are "sample 1" and "sample 2."
       """)
    significance = float(st.text_input('Input significance value (default value is .05)', value='.05'))
    
    uploaded = st.file_uploader('Upload your .csv, .xls, or .xlsx file.')
    st.caption("📝 This app does not retain user data.")
    if uploaded != None:
        if uploaded.name.endswith('csv'):
            df = pd.read_csv(uploaded)
            s1 = [int(c) for c in df['sample 1']]
            s2 = [int(c) for c in df['sample 2']]
       
            chi, p_val, dof, ex = chi2_contingency([s1,s2], correction=False)
            p = 1 - significance
            crit_val = scipy_chi2.ppf(p, dof)
            
            st.subheader('Results')
            st.write('Uploaded Contingency Table:')
            st.write(df)
            c1 = st.container()
            c2, c3, c4, c5 = st.columns(4)
            
            
            c1.metric('p-value', str(p_val))
            c2.metric('# of Samples',str(len(s1)))
            c3.metric('Degrees of Freedom',"{:.2f}".format(dof)) 
            c4.metric('\n chi2 test statistic',"{:.5f}".format(chi)) 
            c5.metric('critical value',"{:.5f}".format(crit_val))
            st.write("For an extended discussion of using chi2 tests for homogeneity for qualitative coding, see [Geisler and Swarts (2019)](https://wac.colostate.edu/docs/books/codingstreams/chapter9.pdf)")
        elif uploaded.name.endswith('xlsx'):
            df = pd.read_excel(uploaded)
            s1 = [int(c) for c in df['sample 1']]
            s2 = [int(c) for c in df['sample 2']]
       
            chi, p_val, dof, ex = chi2_contingency([s1,s2], correction=False)
            p = 1 - significance
            crit_val = scipy_chi2.ppf(p, dof)

            st.subheader('Results')
            st.write('Uploaded Contingency Table:')
            st.write(df)

            c1 = st.container()
            c2, c3, c4, c5 = st.columns(4)
    
            c1.metric('p-value', str(p_val))
            c2.metric('# of Samples',str(len(s1)))
            c3.metric('Degrees of Freedom',"{:.2f}".format(dof)) 
            c4.metric('\n chi2 test statistic',"{:.5f}".format(chi)) 
            c5.metric('critical value',"{:.5f}".format(crit_val))
            st.write("For an extended discussion of using chi2 tests for homogeneity for qualitative coding, see [Geisler and Swarts (2019)](https://wac.colostate.edu/docs/books/codingstreams/chapter9.pdf")
    st.markdown("""### Sample Python Code""")
    with st.expander('Sample Python Code'):
            st.code("""from scipy.stats import chi2_contingency

    sample1 = [35,45,100]
    sample2 = [45,40,200]
    chi, p_val, dof, ex = chi2_contingency([sample1,sample2], correction=False)

    alpha = 0.05
    p = 1- alpha

    critical_value = scipy_chi2.ppf(p, dof)
                    """)
            st.write("For an extended discussion of using chi2 tests for homogeneity for qualitative coding, see [Geisler and Swarts (2019)](https://wac.colostate.edu/docs/books/codingstreams/chapter9.pdf)")
    
          
    
def kappa():
    st.title("Cohen's Kappa Calculator")
    st.markdown("""
Cohen's Kappa offers what Geisler and Swarts (2019) call "corrected agreement" between coders. Corrected agreement accounts for the possibility that coincidence in coding decisions may be the result of random chance than codebook instructions. Cohen's kappa adjusts for chance selection
                """)
    st.latex(r'''
             \kappa = \frac{p_o - p_e}{N-p_e}
             ''')
    st.latex(r'''\kappa = \textrm{Kappa statistic}\newline
 p_o = \textrm{relative observed agreement among coders}\newline
 p_e = \textrm{probability of chance agreement among coders}\newline
 N = \textrm{number of coding decisions}\newline
             
    ''')
    
    
    st.markdown("""
To derive the expected probability (e_p) that a coding decisions could be the result of chance you derive the joint probability of selection for each coding category and then multiply the results by the number of coding decisions. You then sum the expected probability of each coding decisios to derive p_e.

Consider the following coding matrix of agreements and disagreements:""")

    
    st.dataframe(pd.DataFrame(confusion_matrix('pos neg neg neg neg neutral neutral'.split(),'pos neg neg neg neutral neutral neutral'.split()),index=['neg','neutral','pos'],columns=['neg','neutral','pos']))
    st.markdown("""Coder 1 selected _pos_ 1 out of 7 times. Coder 2 also selected _pos_ 1 out of 7 times. The expected probability of _pos_ is:
    
    (1/7) * (1/7) * 7 = 0.143 = Expected probability of _pos_
    
For _neutral_:
    
    (3/7) * (2/7) * 7 = 0.857 = Expected probability of _neutral_
    
For _neg_:
    
    (3/7) * (4/7) * 7 = 1.714 = Expected probability of _neg_
    
    Probability of chance agreement = (0.143 + 0.857 + 1.714) = 2.714
    
In other words:
    
    kappa = (6-2.714)/(7-2.714) = .7667
                """)
                
                
    st.markdown("""
### When to use Cohen's Kappa?
Cohen's Kappa is used when determining the inter-rater agreement of 2 raters. According to Geisler and Swarts (2019), Cohen's Kappa is best suited for balanced datasets (e.g., coding decisions that are evenly distributed). For skewed coding samples, see Krippendorff's Alpha.                
                """)
    st.markdown("""
### How to Interpret Cohen's Kappa Statistic?
Cohen's Kappa statistics range from 0 - 1 with 0 indicating perfect disagreement and 1 indicating perfect agreement. Scores greater than or equal to .8 indicate high inter-rater reliability. Scores less than .677 indicate low agreement and reliability.
                """)
                
    st.markdown("For more an extended presentation on Cohen's Kappa see Hart-Davidson (2014), [Using Cohen's Kappa to Gauge Interrater Reliability(https://www.slideshare.net/billhd/kappa870)")
    st.write("""
    1. Copy the codes of Coder 1 into the Coder 1 text entry field and hit "Enter." 
    2. Copy the codes for Coder 2 into the Coder 2 text entry field and hit "Enter."
    ❗ Make sure that the coding decisions between Coder 1 and Coder 2 are the same length.
       """)
    
    col1 = st.text_input('Coder 1',value='a a b')
    col2 = st.text_input('Coder 2', value='a a b')
    st.caption("📝 This app does not retain user data.")
 
    try:
        st.subheader('Results')
        c1, c2, c3 = st.columns(3)
        c1.metric('Dataset Length',str(len(col1.split())))
        c2.metric('Accuracy',str(accuracy_score(col1.split(),col2.split())))
        c3.metric('Kappa Score',str(cohen_kappa_score(col1.split(),col2.split())))

        labels = sorted(list(set(col1.split()+ col2.split())))
        indices = [str(label)+'_' for label in labels]
        st.write("Confusion Matrix:")
        st.dataframe(pd.DataFrame(confusion_matrix(col1.split(),col2.split()),index=indices,columns=labels))
        st.caption('Note: Coder 1 is used as the baseline for evaluation.')
        
    except ValueError:
        st.markdown('<mark>Error: Data must be the same length</mark>', unsafe_allow_html=True)
    st.markdown("""### Sample Python Code""")
    with st.expander('Sample Python Code'):
        st.code("""from sklearn.metrics import cohen_kappa_score as kappa
                
sample1 = ['D','E','D','S','D']
sample2 = ['D','E','S','D','D']

kappa_score = kappa(sample1,sample2)

[Out]: 0.28571
                """)
        st.markdown("For more an extended presentation on Cohen's Kappa see Hart-Davidson (2014), [Using Cohen's Kappa to Gauge Interrater Reliability](https://www.slideshare.net/billhd/kappa870)")
        
from nltk.metrics.agreement import AnnotationTask    
def k_alpha():
    st.title("Krippendorff's Alpha Calculator")  
   
    st.markdown("""
Krippendorff's Alpha provides what Geisler and Swarts (2019) call "corrected agreement" between coders. Corrected agreement accounts for the possibility that coincidence in coding decisions may be the result of random chance than codebook instructions. Krippendorff's alpha corrects for chance by dividing observed disagreements by expected disagreements and subtracting from 1. 

Krippendorff's Alpha is given by:
    """)
    st.latex(r'''
             \alpha = 1 - D_O/D_E 
             ''')
    
    
    st.latex(r'''\alpha = \textrm{alpha statistic}\newline
 D_O = \textrm{observed disagreement}\newline
 E_E = \textrm{expected disagreement}\newline
 ''')
    st.write("Krippendorff's alpha is calculating using `The Natural Language Toolkit` (Bird, Loper, and Klein, 2009")
    st.markdown("""### When To Use Krippendorff's Alpha for Inter-rater Reliability?

Following Geisler and Swarts (2109, p. 167-168), Krippendorff's Alpha works best when coding decisions are skewed towards a select number of coding categories. Cohen's Kappa, the other inter-rater reliability metric discussed in Geisler and Swarts (2019), is better for more balanced coding samples. 
                """)
                
    st.markdown("""
### How to interpret Krippendorff's Alpha?

Krippendorff alpha scores range from 0 - 1 wth 0 indicating perfect disagreement and 1 indicating perfect agreement. Scores greater than or equal to .80 indicate high rater agreement. Scores lower than .677 indicate suboptimal agreement or agreement that is influenced by chance at greater than acceptable rates.
                 """)
    
    st.markdown("""
### Input your data""")
    st.write("""
    1. Copy the codes of Coder 1 into the Coder 1 text entry field and hit "Enter." 
    2. Copy the codes for Coder 2 into the Coder 2 text entry field and hit "Enter."
    ❗ Make sure that the coding decisions between Coder 1 and Coder 2 are the same length.
       """)
    
    col1 = st.text_input('Coder 1',value='a a b')
    col2 = st.text_input('Coder 2', value='a a b')
    
    col_data = []
    for i in range(len(col1)):
        col_data.append(('c1',i,col1[i])) 
        col_data.append(('c2',i,col2[i]))
    
    t = AnnotationTask(data=col_data)
    
    
    st.caption("📝 This app does not retain user data.")
    
    
 
    try:
        st.subheader('Results')
        c1, c2, c3 = st.columns(3)
        c1.metric('Dataset Length',str(len(col1.split())))
        c2.metric('Accuracy',str(accuracy_score(col1.split(),col2.split())))
        c3.metric('Alpha',str((t.alpha())))

        labels = sorted(list(set(col1.split()+ col2.split())))
        indices = [str(label)+'_' for label in labels]
        st.write("Confusion Matrix:")
        st.dataframe(pd.DataFrame(confusion_matrix(col1.split(),col2.split()),index=indices,columns=labels))
        st.caption('Note: Coder 1 is used as the baseline for evaluation.')
    except ValueError:
        st.markdown('<mark>Error: Data must be the same length</mark>', unsafe_allow_html=True)
    st.markdown("""### Sample Python Code""")
    with st.expander('Sample Python Code'):
        st.code("""from nltk.metrics.agreement import AnnotationTask

#calculate Krippendorff's Alpha
col1 = ['a','a','b']
col2 = ['a','b','b']

#format data for NLTK AnnotationTask
#NLTK AnnotationTask format [('coder1',index,code),('coder2',index,code) . . .]

col_data = []
for i in range(len(col1)):
    col_data.append(('c1',i,col1[i])) 
    col_data.append(('c2',i,col2[i]))

t = AnnotationTask(data=col_data)

#print Krippendorff's Alpha
print(t.alpha())

[Out]: 0.7272
                """)
def k_alpha_file_upload():
    st.title("Krippendorff's Alpha Calculator")  
    
    st.markdown("""
Krippendorff's Alpha provides what Geisler and Swarts (2019) call "corrected agreement" between coders. Corrected agreement accounts for the possibility that coincidence in coding decisions may be the result of random chance than codebook instructions. Krippendorff's alpha corrects for chance by dividing observed disagreements by expected disagreements and subtracting from 1. 

Krippendorff's Alpha is given by:
    """)
    st.latex(r'''
             \alpha = 1 - D_O/D_E 
             ''')
    
    st.latex(r'''\alpha = \textrm{alpha statistic}\newline
 D_O = \textrm{observed disagreement}\newline
 E_E = \textrm{expected disagreement}\newline
 ''')
             

    st.markdown("""### When To Use Krippendorff's Alpha for Inter-rater Reliability?

Following Geisler and Swarts (2109, p. 167-168), Krippendorff's Alpha works best when coding decisions are skewed towards a select number of coding categories. Cohen's Kappa, the other inter-rater reliability metric discussed in Geisler and Swarts (2019), is better for more balanced coding samples. 
                """)
                
    st.markdown("""
### How to interpret Krippendorff's Alpha?

Krippendorff alpha scores range from 0 - 1. Scores greater than or equal to .80 indicate high rater agreement. Scores lower than .677 indicate suboptimal agreement or agreement that is influenced by chance at greater than acceptable rates.

                 """)
    st.write("Krippendorff's alpha is calculating using `The Natural Language Toolkit` (Bird, Loper, and Klein, 2009")
    st.markdown("""
    ### Upload your .csv, .xls, or .xlsx file. 
    
    Your files should feature the following format:
       """)
       
    dff = pd.DataFrame({'Coder 1':['a','a','b'],'Coder 2': ['a','a','b']})
    st.dataframe(dff)
    
    uploaded_file = st.file_uploader("Upload your data as .csv or .xlsx")
    st.caption("📝 This app does not retain user data.")
    
    col1 = dff['Coder 1'].tolist()
    col2 = dff['Coder 2'].tolist()
    


    col_data = []
    for i in range(len(col1)):
        col_data.append(('c1',i,col1[i])) 
        col_data.append(('c2',i,col2[i]))
    
    t = AnnotationTask(data=col_data)

 
    st.subheader('Results')
   
    c1, c2, c3 = st.columns(3)
    c1.metric('Dataset Length',str(len(col1)))
    c2.metric('Accuracy',str(accuracy_score(col1,col2)))
    c3.metric('Alpha',str((t.alpha())))

    labels = sorted(list(set(col1+ col2)))
    indices = [str(label)+'_' for label in labels]
    st.write("Confusion Matrix:")
    st.dataframe(pd.DataFrame(confusion_matrix(col1,col2),index=indices,columns=labels))
    st.caption('Note: Coder 1 is used as the baseline for evaluation.')
    if uploaded_file != None:
        if str(uploaded_file.name).endswith('csv'):
            df = pd.read_csv(uploaded_file,encoding_errors='ignore',encoding='latin1')

            st.subheader('Results')
         
            col1 = df['Coder 1'].tolist()
            col2 = df['Coder 2'].tolist()
            
    
    
            col_data = []
            for i in range(len(col1)):
                col_data.append(('c1',i,col1[i])) 
                col_data.append(('c2',i,col2[i]))
            
            t = AnnotationTask(data=col_data)
    
    
            st.caption("📝 This app does not retain user data.")
         
            try:
                st.subheader('Results')
                c1, c2, c3 = st.columns(3)
                c1.metric('Dataset Length',str(len(col1.split())))
                c2.metric('Accuracy',str(accuracy_score(col1.split(),col2.split())))
                c3.metric('Alpha',str((t.alpha())))
        
                labels = sorted(list(set(col1.split()+ col2.split())))
                indices = [str(label)+'_' for label in labels]
                st.write("Confusion Matrix:")
                st.dataframe(pd.DataFrame(confusion_matrix(col1.split(),col2.split()),index=indices,columns=labels))
                st.caption('Note: Coder 1 is used as the baseline for evaluation.')
            except ValueError:
                st.markdown('<mark>Error: Data must be the same length</mark>', unsafe_allow_html=True)
        elif str(uploaded_file.name).endswith('xlsx'):
            try:
                st.subheader('Results')
                c1, c2, c3 = st.columns(3)
                c1.metric('Dataset Length',str(len(col1.split())))
                c2.metric('Accuracy',str(accuracy_score(col1.split(),col2.split())))
                c3.metric('Alpha',str((t.alpha())))
        
                labels = sorted(list(set(col1.split()+ col2.split())))
                indices = [str(label)+'_' for label in labels]
                st.write("Confusion Matrix:")
                st.dataframe(pd.DataFrame(confusion_matrix(col1.split(),col2.split()),index=indices,columns=labels))
                st.caption('Note: Coder 1 is used as the baseline for evaluation.')
            except ValueError:
                st.markdown('<mark>Error: Data must be the same length</mark>', unsafe_allow_html=True)
    st.markdown("""### Sample Python Code""")
    with st.expander('Sample Python Code'):
        st.code("""from nltk.metrics.agreement import AnnotationTask

    #calculate Krippendorff's Alpha
    col1 = ['a','a','b']
    col2 = ['a','b','b']

    #format data for NLTK AnnotationTask
    #NLTK AnnotationTask format [('coder1',index,code),('coder2',index,code) . . .]
    col_data = []
    for i in range(len(col1)):
        col_data.append(('c1',i,col1[i])) 
        col_data.append(('c2',i,col2[i]))

    t = AnnotationTask(data=col_data)

    #print Krippendorff's Alpha
    print(t.alpha())
    
    [Out]: 0.7272
                    """)
        
def kappa_file_upload():
    st.title("Cohen's Kappa Calculator")
    st.markdown("""
Cohen's Kappa offers what Geisler and Swarts (2019) call "corrected agreement" between coders. Corrected agreement accounts for the possibility that coincidence in coding decisions may be the result of random chance than codebook instructions. Cohen's kappa adjusts for chance selection
                """)
    st.latex(r'''
             \kappa = \frac{p_o - p_e}{N-p_e}
             ''')
    st.latex(r'''\kappa = \textrm{Kappa statistic}\newline
 p_o = \textrm{relative observed agreement among coders}\newline
 p_e = \textrm{probability of chance agreement among coders}\newline
 N = \textrm{number of coding decisions}\newline
             
    ''')
    
    
    st.markdown("""
To derive the expected probability (e_p) that a coding decisions could be the result of chance you derive the joint probability of selection for each coding category and then multiply the results by the number of coding decisions. You then sum the expected probability of each coding decisios to derive p_e.

Consider the following coding matrix of agreements and disagreements:""")

    
    st.dataframe(pd.DataFrame(confusion_matrix('pos neg neg neg neg neutral neutral'.split(),'pos neg neg neg neutral neutral neutral'.split()),index=['neg','neutral','pos'],columns=['neg','neutral','pos']))
    st.markdown("""Coder 1 selected _pos_ 1 out of 7 times. Coder 2 also selected _pos_ 1 out of 7 times. The expected probability of _pos_ is:
    
    (1/7) * (1/7) * 7 = 0.143 = Expected probability of _pos_
    
For _neutral_:
    
    (3/7) * (2/7) * 7 = 0.857 = Expected probability of _neutral_
    
For _neg_:
    
    (3/7) * (4/7) * 7 = 1.714 = Expected probability of _neg_
    
    Probability of chance agreement = (0.143 + 0.857 + 1.714) = 2.714
    
In other words:
    
    kappa = (6-2.714)/(7-2.714) = .7667
                """)
                
                
    st.markdown("""
### When to use Cohen's Kappa?
Cohen's Kappa is used when determining the inter-rater agreement of 2 raters. According to Geisler and Swarts (2019), Cohen's Kappa is best suited for balanced datasets (e.g., coding decisions that are evenly distributed). For skewed coding samples, see Krippendorff's Alpha.                
                """)
    st.markdown("""
### How to Interpret Cohen's Kappa Statistic?
Cohen's Kappa statistics range from 0 - 1 with 0 indicating perfect disagreement and 1 indicating perfect agreement. Scores greater than or equal to .8 indicate high inter-rater reliability. Scores less than .677 indicate low agreement and reliability.
                """)
    st.markdown("For more an extended presentation on Cohen's Kappa see Hart-Davidson (2014), [Using Cohen's Kappa to Gauge Interrater Reliability(https://www.slideshare.net/billhd/kappa870)")
    st.markdown("""
    ### Upload your .csv, .xls, or .xlsx file. 
    
    Your files should feature the following format:
       """)
       
    dff = pd.DataFrame({'Coder 1':['a','a','b'],'Coder 2': ['a','a','b']})
    st.dataframe(dff)
    
    uploaded_file = st.file_uploader("Upload your data as .csv or .xlsx")
    st.caption("📝 This app does not retain user data.")
    if uploaded_file != None:
        if str(uploaded_file.name).endswith('csv'):
            df = pd.read_csv(uploaded_file,encoding_errors='ignore',encoding='latin1')

            st.subheader('Results')
         
            col1 = df['Coder 1'].tolist()
            col2 = df['Coder 2'].tolist()
     
            c1, c2, c3 = st.columns(3)
            c1.metric('Dataset Length',str(len(col1)))
            c2.metric('Accuracy',str(accuracy_score(col1,col2)))
            c3.metric('Kappa Score',str(cohen_kappa_score(col1,col2)))

            labels = sorted(list(set(col1+ col2)))
            indices = [str(label)+'_' for label in labels]
            st.write("Confusion Matrix:")
            st.dataframe(pd.DataFrame(confusion_matrix(col1,col2),index=indices,columns=labels))
            st.caption('Note: Coder 1 is used as the baseline for evaluation.')

            
            
        #except ValueError:
             #   st.markdown('<mark>Error: Data must be the same length</mark>', unsafe_allow_html=True)
        elif str(uploaded_file.name).endswith('xls'):
           df = pd.read_excel(uploaded_file)
           col1 = df['Coder 1'].tolist()
           col2 = df['Coder 2'].tolist()

           st.subheader('Results')
           c1, c2, c3 = st.columns(3)
           c1.metric('Dataset Length',str(len(col1)))
           c2.metric('Accuracy',str(accuracy_score(col1,col2)))
           c3.metric('Kappa Score',str(cohen_kappa_score(col1,col2)))

           labels = sorted(list(set(col1+ col2)))
           indices = [str(label)+'_' for label in labels]

           st.write("Confusion Matrix (Coder 1 is treated as the baseline for evaluation):")
           st.dataframe(pd.DataFrame(confusion_matrix(col1,col2),index=indices,columns=labels))
           st.caption('Note: Coder 1 is used as the baseline for evaluation.')

           st.markdown("For a more extensive presentation on Cohen's Kappa see Hart-Davidson (2014), [Using Cohen's Kappa to Gauge Interrater Reliability](https://www.slideshare.net/billhd/kappa870)")
           #except ValueError:
            #   st.markdown('<mark>Error: Data must be the same length</mark>', unsafe_allow_html=True)
    st.markdown("""### Sample Python Code""")
    with st.expander('Sample Python Code'):
        st.code("""from sklearn.metrics import cohen_kappa_score as kappa
                
sample1 = ['D','E','D','S','D']
sample2 = ['D','E','S','D','D']

kappa_score = kappa(sample1,sample2)

[OUT]: .28571
                """)
        st.markdown("For more an extended presentation on Cohen's Kappa see Hart-Davidson (2014), [Using Cohen's Kappa to Gauge Interrater Reliability](https://www.slideshare.net/billhd/kappa870)")
def lgr_classify_table():
    st.title('Multinomial Logistic Regression and Classification')
    st.markdown("""The goal of logistic regression, as Geisler and Swarts (2019) and Massaron and Boschetti (2016) point out is **prediction**. For regression tasks, this means regressing independent variables to a categorical response or predictor variable to determine the influence of independent variables on the outcomes variables.

Logistic regression classification is another linear model, similar to linear regression (Geisler and Swarts, 2019; Massaron and Boschetti, 2016) in that data are fitted to a line. However, in the case of logistic regression, data are fitted to a logistic function, which takes the form of an S-Curve:""")
    np.random.seed(37)
    warnings.filterwarnings('ignore')

    x = np.arange(-6, 6.1, 0.1)
    y = logistic(x)

    fig, ax = plt.subplots(figsize=(15, 6))
    _ = ax.plot(x, y,c='b')
    _ = ax.set_title('S-curve plot of logistic function')
    
    st.pyplot(fig)
    st.caption("S-curve plot of logistic function. Chart code sourced from One-Off Coder, 2019.")
                
    st.markdown("""The logistic function models the probability that a data point belongs to a class. Consequently, data that falls closer to 1 will be assigned to the positive class (1) and data that falls closer to 0 will be assigned to the negative class (0). 
                
For classification tasks, logistic regressor models would assign a class label (response variable) to data (independent variables) based on likelihood (see McNulty, 2021 for a discussion of the work logistic regressors can do for regression and classification). In the case of binary classification (classes 0 and 1), a binomial logistic regression classifier model would output probability scores for each class, which total to 1 (see Geisler and Swarts, 2019, pp. 318-319):
    
    #class 0 = negative
    #class 1= positive
    
    classifier.predict("This is the worst movie of all time!")
    
    [Out]: {0:0.76, 1:0.24}
    
    #The most probable class is 0 (negative)
        
    classifier.predict("This is the best move of all time!")
    
    [Out]: {0:0.06, 1:0.94}

    #The most probably class is 1 (positive).
    """)
    


    
    st.markdown("""
                
The probability that observations belong to classes 0 and 1 are given by the following logistic functions (Massaron and Boschetti, 2016; see also McNulty, 2021):
                """)
    st.latex(r'''
             P(y=0|x) =\sigma(W^T*x)\newline
             
             P(y=1|x) =\sigma(W^T*x)\newline
             
             ''')
    st.markdown("""Where P(y=0|x) is the probability of class 0 given x and P(y=1|x)) is the probability of class 1 given x.

The above formulation include binary classification. However, logistic regression can be extended to the multinomial case (3 or more classes).             
                """)
    st.markdown("""### When to Use Multinomial Logistic Regression?

You can use the logistic regression when you wish to model the influence of independent variables on 2 or more response variables and/or predict categorical variables (hence, *multinomial* logistic regression).

Your training/predictor data can be categorical or continuous, although all data must be converted into numerals. If you have categorically labeled data, `QC Utils` will one-hot encode those labels for machine processing. One-hot encoding will convert string labels into numeric representations.

If you wish to create a logistic regression classifier for textual data, use "Logistic Regression Classification Text" instead (textual data requires additional preprocessing to extract features or independent variables).
### How to Run a Multinomial Logistic Regression with `QC Utils`?

- Format your data as n_samples x n_features and label your dependent variable "target":
    
    |f1|f2|f3|target|
    |--|--|--|------|
    |x1|x2|x3|A|
    
    :memo: It doesn't matter what order the columns are, so long as the dependent variable (predicted class) is labeled "target;" however, the name of the column will determine the order of the independent variables.  
- If the independent or predictor variables of your data are categorical, select "One-Hot Encode" to format your data for analysis.
- Upload your data as a .csv, .xls, or .xlsx file.

*Optional*: enter the name of a target value/class label to drop. As discussed above (see also Geisler and Swarts, 2019), logistic regression creates a "reference" model from which model class coefficients are calculated. For examle, in the binary case (classes 0 and 1), the logistic regression model would take at its point of reference class 0 and then calculate the coefficients for class 1. This is why `statsmodels` returns a single columns of coefficients and log odds for binary data. In the multinomial case (classes 0, 1, 2), the model coefficents of 1 and 2 would be derived from the reference class 0. By default, the `statsmodels` API will return all class coefficient information except for the reference class. 

In the multinomial case, you can drop target values in order to select the classes of interest.

`statsmodels` will set as the reference class the class that appears first alphabetically ('a','b','c') or numerically (0,1,2). You can pre-select the reference class processed by `statsmodels` by the adjusting the names of your classes. For example, your "class of interest" in the language of Geisler and Swarts (2019) could be labeled '0.' 

Lastly, we can conduct an analysis of the logistic regression model marginals (Norton et al., 2019) to interpret the influence of independent variables on all classes without dropping class values. `statsmodels` will return a summary report of model marginals. In the next section, I discuss how we can use model marginals for interrogate logistic regression models. 

### How to Interpret the Results of Multinomial Logistic Regression?

Geisler and Swarts (2019) use the log-odds of the logistic regression model to explain the chance that an independent variable will influence a depdendent variable/predicted outcome. The log-odds are the natural logarithm of the model odds, i.e., the odds that a given independent variable will contribute to an outcome. In this section, I describe how to use odds, odds ratios, log odds, and probability to understand a logistic regression model.  

##### Odds, Odds Ratios, and Probability Scores
Consider the following table of data:

||roses sold|roses unsold|
|--|--------|------------|
|yellow|10|40|
|red|40|20|

The odds that the roses sold are yellow is 10/40 or .25. We can convert odds to a probability by:
        odds/1+odds
        
The probability that the roses sold are yellow is .25/1+.25 or .20 or 20%.

The odds that yellow roses are unsold are 40/10 or 4. The probability that the yellow roses will be unsold is 4/1+4 or .80 or 80% (as we would expect since the probability of the unsold is 1 - the probability of yellow roses sold).

We can calculate the *odds ratio* of yellow roses sold to unsold by dividing the odds of yellow-sold by the odds of yellow-unsold or .25/4 or .0625. The odds ratio of yellow-unsold to yellow-sold is 4/.25 or 16. Odds ratios close to or equal to 1.0 indicate that there is no difference in odds between variables. Odds ratios greater than 1.0 indicate that the first case is more likely to occur or increase. Values less than 1.0 indicate that the variable is less predicative of the outcome. 

If we compare which color of rose is most likely to be sold, then the odds for yellow are 10/40 or .25. The odds that red roses will be sold are 40/10 or 4. The odds ratio for red to yellow roses is 4/.25 or 16. The odds ratio yellow to red roses sold is .25/4 or .0625. This indicates that among roses sold, there is a far greater chance that the rose will be red (for a discussion of odds ratios, see Szumilas, 2010; Greenfield et al., 2008).

The log-odds of the logistic regression model are the natural logarithms of the odds ratios. Thus, the log odds provide similar information about the model. The log-odds indicate the magnitude of influence of the independent variable on the dependent/response variable. 

`QC Utils` returns the odds ratios, log-odds, and probability scores for the logistic regression model in its reporting.

##### Marginal Effects 
Following Norton et al. (2019) and Cummings et al. (2013), we can use the marginal effects of each independent variable per class/outcome. Norton et al. (2019, p. 1305) states that "Marginal effects are a useful way to describe the average effect of changes in explanatory variables on the change in the probability of outcomes." In other words, the marginal effects of an independent variable illustrate the magnitude of influence that variable supplies as the probability of the outcome increases. For example, if the variable `alcohol` has a marginal effect of -0.0016 on class_0, this means that the influence of alcohol decreases the more likely the outcome is class_0. On the other hand, if the marginal effect of `alcohol` on class_1 is 0.0080, then the influence of alcohol increases as the probability of class_1 nears 1.

The `statsmodels` API provides us with a report of marginal effects. This report includes information on all model classes/response variables unlike the `statsmodels` summary report, allowing us a convenient means to compare the effects of independent variables among model classes/response variables. 

##### p-values
Odds ratios, log-odds, probability scores, and marginal effects indicate the predictive influence of independent variables on the response variable/class outcome. However, not all independent variables are signficant explainers of the outcome. We can determine the significance of an independent variable to a class outcome by evaluating its p-value. P-values indicate the degree of confidence we can claim that the observations are not the product of chance based on a significance threshold (alpha value). An alpha value of .05 sets the threshold at a 95% confidence level (1 - alpha). If an independent variable has a p-value less than .05, then we can conclude with 95% confidence that the variable is significant to the model. If the p-value is higher than 0.05, then we can conclude that the variable is not significant. 

The `statsmodels` API returns p-values in its summary and marginal effects reports. Identifying the significance of a model's independent variables can focus attention on the most influential features of the model and be used for model tuning when mounting classification tasks--insignificant independent variables can be removed from the training set as noise, which, ideally, would lead to improved classification accuracy.

The `statsmodels` API also outputs a "LLR p-value" for the entire model. The LLR p-value indicates the likelihood that the model fit is better or worse than the null model--the logistic regression model with no independent variables, only constants. In other words, LLR p-value tells us whether or not the observed variables are more explanatory to model outcomes than chance. The LLR p-value of the wine dataset example below is 2.065e-37, which is a miniscule value well below 0.05. Thus, the model is better fitted than the null and the independent variables under the model are significant.

##### McFadden's (1979) pseudo R-squared
`statsmodels` includes a psuedo R-squared statistics in its report of summary statistics. The pseudo R-squared is derived from McFadden (1979) and indicates the fit of the observed model compared to a null model (the logistic regression model without independent variables). Pseudo R-squared scores above 0.40 are considered good models fits (see Hemmert et al., 2016).

The demo logistic regression model trainined on `sklearn's` wine dataset yields a pseudo R-squared of 0.79 (see below), which would be considered a good fit. 
                """)
    encoder_option = st.checkbox("One-Hot Encode")
    uploaded_file = st.file_uploader("Upload your data as .csv, .xls, or .xlsx")
    st.caption("📝 This app does not retain user data.")
    if uploaded_file != None:
        train_lgr_table(uploaded_file, encoder_option)
    else:
        lgr_demo_tabular()
    st.markdown("""### Sample Python Code""")
    with st.expander("Sample Python Code"):
        st.markdown("""#### Continuous Independent Variables""")
        st.code("""from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm

#load and make training and testing data
wine = load_wine(as_frame=True)
X_train, x_test, y_train, y_test = train_test_split(wine.data,wine.target)

#add constant to training data (required by statsmodels and logistic regression)
train_data = pd.DataFrame(sm.add_constant(X_train,has_constant='add'))


#instantiate statsmodels.discreet.discreet.Logit model class with training and target data
logit_mod = sm.MNLogit(y_train,train_data)

#fit the logitic regression model
logit_res = logit_mod.fit()

#print model summary statistics
st.write(logit_res.summary())
                """)
        st.markdown("""#### Categorical Independent Variables
Logistic regression requires numerical data. If your independent variables are categorical (as illustrted in Geisler and Swarts (2019)), then categorical data must be converted using one-hot encoding. 
                    """)
                    
        st.code("""from sklearn.preprocessing import OneHotEncoder

#instantiate OneHotEncoder
enc = OneHotEncoder()

#data
var1 = 'D E E D D D S D E E D D D S D E E D D D S D E E D D D S D E E D D D S D E E D D D D'.split()
var2 = 'neu pos pos neu neu neg neu neu pos pos neu neu neg neu neu pos pos neu neu neg neu neu pos pos neu neu neg neu neu pos pos neu neu neg neu neu pos pos pos neu neg neg'.split()

y_ = 'bio bio bio bio eng eng eng bio bio bio bio eng eng eng bio bio bio bio eng eng eng bio bio bio bio eng eng eng bio bio bio bio eng eng eng bio bio bio bio eng eng eng'.split()



df = pd.DataFrame()
df['var1'] = var1
df['var2'] = var2
df['target'] = y_


train = enc.fit_transform(df[['var1','var2']]).todense()

#add a column on 1.0 to create an intercept as required by logistic regression
train = sm.add_constant(train, has_constant='add')

lb = LabelBinarizer()
targets = lb.fit_transform(df['target'])
#transform data
logit_mod = sm.MNLogit(targets,train)
logit_res = logit_mod.fit()

#print model summary statistics
logit_res.summary()
                """)
        
def lnr_fit():
    st.title('Linear Regression')
    
    st.markdown("""
                
Linear regression attempts to model the relationship between variables in a line of fit that reduces the sum of squared residuals between observed and predicted values. In the simplest form of linear regression, there are two variables: independent and dependent. A linear regression model attempts to predict the value of the dependent variable as a result of the linear combination of the independent variable.

This relationship can be captured in the slope of x and y values in 2-dimensions:""")

    st.latex(r'''
             y = ax + b
             ''')
    
    st.markdown("""Here a refers to the _slope_ of the line; b refers to the _intercept_. Thus, _y_, the dependent variable is a function of the line of fit produced by the linear combination the independent _x_ variable.
                
As VanderPlas (2016) notes, linear regression can be applied to multiple dimensions and include more complex lines of fit. In the case when there is more than one independent variable, the linear combination would proceed as (formula quoted from VanderPlas, 2016):""")

    st.latex(r'''
             y = a_0x_0 + a_1x_1 + a_2x_2 \textrm{. . .} \newline
             ''')
    st.markdown("""### When to Run a Linear Regression?
Linear regression is suitable when modeling continuous variables and when your independence variables are independent from each other. For example, the demo linear regression model used in `QC Utils` is the California Housing Dataset (Pace and Barry, 1997). The dependent or target variable is median home values. Independent variables include latitude and longitude of homes, the number of rooms, median income, and average bedrooms.

Linear regression analysis can be used to determine the contribution of the independent variable(s) to the dependent variable and to make predictions on new (continuous) data.

                """)
                
    

    st.markdown("""### How to Interpret Linear Regression Results?
The linear regression model  can be used to explain the influence of each independent variable on the dependent variable and predict target values (i.e., median home values).

We can use the attribute of a fitted linear regression model to better understand the influence of each independent variable by examining the model coefficients.

Examining the linear regression model example fitted on the California Housing Dataset (Pace and Barry, 1997; see Linear Regression Demo below), we see that the dependent/target variable is "median_home_value." One independent variable is "MedInc" (median income) with a coefficient of .4487. This means that with every unit increase in the dependent variable (median housing value), there is a .4487 increase in median income. The independent variable of "AveBedrms" (average bedrooms) has a coefficient of .7831. This means that for every one unit increase in median house value, there is a .7831 increase in the average number of bedrooms.

The `statsmodels` linear regression results output also indicates the p-value of each independent variable. Values lower than .01 indicate that the independent variable makes a significant contribution to the value of the dependent variable. 

The R<sup>2</sup> indicate the proportion of the variance explained by the independent variables. In the case of the California Housing Dataset, the R<sup>2</sup> is .613. This means that 61.3 of the variance in the dataset can be explained by the independent variables. 

We can also use fitted linear regression models to make predictions. The line chart in the Linear Regression demo below depicts predictions (red) versus observed values (blue), giving us another view of the strength of the model.

                """,unsafe_allow_html=True)
    st.markdown("""### How to Run a Linear Regression?

- Upload your data as a .csv, .xls or .xlsx file

The rows of your data should be cases. The columns should contain your variables. Your dependent variable name should be "target." 

For example:
    
|var1|var2|var3|target|
|----|----|----|------|
|.38|.667|1.9|3|
|.5|.88|2.1|4
|.34|.76|1.8|3.5|

                """)
    uploaded_file = st.file_uploader("Upload your data as .csv, .xls, or .xlsx")
    st.caption("📝 This app does not retain user data.")
    if uploaded_file != None:
        train_lnr_table(uploaded_file)
    else:
        lnr_demo()
    st.markdown("""### Sample Python Code""")
    with st.expander("Sample Python Code"):
        st.code("""import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.20, random_state=42)

#add the required constant
housing_exog = sm.add_constant(X_train, prepend=False)
        
#fit the linear regression model
mod = sm.OLS(y_train, housing_exog)
res = mod.fit()

#add constant to X_test and predict
X_test_exog = sm.add_constant(X_test,prepend=False)
predictions = res.predict(X_test_exog)
        
#plot observed versus predictions with matplotlib
fig, ax = plt.subplots()
ax.plot(range(len(X_test[:200])),y_test[:200],color='b')
ax.plot(range(len(X_test[:200])),predictions[:200],color='r')
ax.legend(['Observed','Predicted'])
plt.title('Linear Regression Plot of California Housing Data')
st.pyplot(fig)
    
#print model summary with statsmodels API
print(res.summary())
        """)

def lnr_demo():
    st.subheader("Linear Regression Demo")
    st.markdown("""This demo uses Pace and Barry's (1997) California Housing Dataset to train a linear regression model.

The linear regression model is trained on 8 features: Median Income by block, House Age, Average Rooms, Average Bedrooms ,Population, Average Occupany, Latitude, and Longitude (independent variables). The dependent variable is Median House Value.
""")
    #lnr = LinearRegression()

    housing = fetch_california_housing(as_frame=True)
    with st.expander("Exapnd to read more about the California Housing Dataset Description (from sklearn API)"):
        st.write(housing.DESCR)
        st.markdown("### Sample Housing Data")
        st.write(housing.data.head()) 
    X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.20, random_state=42)
    #lnr.fit(X_train, y_train)
    housing_exog = sm.add_constant(X_train, prepend=False)
    mod = sm.OLS(y_train, housing_exog)
    res = mod.fit()

    X_test_exog = sm.add_constant(X_test,prepend=False)
    predictions = res.predict(X_test_exog)
    
    st.markdown("""### Results""")

    fig, ax = plt.subplots()
    ax.plot(range(len(X_test[:200])),y_test[:200],color='b')
    ax.plot(range(len(X_test[:200])),predictions[:200],color='r')
    ax.legend(['Observed','Predicted'])
    plt.title('Linear Regression Plot of California Housing Data')
    st.pyplot(fig)
    st.caption("Linear regression plot of the first 200 observed and predicted values of the California Housing Dataset (Pace and Barry, 1997)")
    
    st.markdown('#### Summary Statistics (provided by `statsmodels` API)')
    
    st.write(res.summary())

   
        
    

def lgr_classify_text():
    st.title('Logistic Regression Classification of Textual Data')
    st.markdown("""
Logistic regression classification is another linear model, similar to linear regression (Geisler and Swarts, 2019; Massaron and Boschetti, 2016). The goal of logistic regression, as Geisler and Swarts (2019) and Massaron and Boschetti (2016) point out is **prediction**. For regression tasks, this means regressing independent variables to a categorical response or predictor variable to determine the influence of independent variables on the outcomes variables. 
                
For classification tasks, logistic regressor models would assign a class label (response variable) to data (independent variables) based on likelihood (see McNulty, 2021 for a discussion of the work logistic regressors can do for regression and classification). In the case of binary classification (classes 0 and 1), a binomal logistic regression classifier model would output probability scores for each class, which total to 1:
    
    #class 0 = negative
    #class 1= positive
    
    classifier.predict("This is the worst movie of all time!")
    
    [Out]: {0:0.76, 1:0.24}
    
    #The most probable class is 0 (negative)
        
    classifier.predict("This is the best move of all time!")
    
    [Out]: {0:0.06, 1:0.94}

    #The most probably class is 1 (positive).
    
The probability that observations belong to classes 0 and 1 are given by the following logistic functions (Massaron and Boschetti, 2016; see also McNulty, 2021):
                """)
    st.latex(r'''
             P(y=0|x) =\sigma(W^T*x)\newline
             
             P(y=1|x) =\sigma(W^T*x)\newline
             
             ''')
    st.markdown("""Where P(y=0|x) is the probability of class 0 given x and P(y=1|x)) is the probability of class 1 given x.

The above formulation include binary classification. However, logistic regression can be extended to the multinomial case (3 or more classes).        
In their discussion of coding textual data, Geisler and Swarts (2019, p. 113) raise the possibility of qualitative coding for machine learning applications. `QC Utils` logistic regression classifier demonstrates how to employ coded textual data for machine classification.
    
The process involves:
 - inputing labeled textual data to the classifier pipeline
 - converting the raw textual data into weighted term vectors
 - training a logistic regression classifier with weighted term vectors and labels
    
#### Text Processing

Text processing involves converting raw textual data (i.e., strings) into numerical representations. The simplest numerical representation is frequency counts: we represent each labeled text as a matrix of counts per each term. This matrix representation includes the entire vocabulary of the corpus. Consider the following corpus of two sentences: "this is a test sentence" and "this is another test sentence"
    
A document-term frequency matrix would resemble the following:
        
|sentence|a|another|is|sentence|test|this|
|--------|-|-------|--|--------|----|----|
|s1|1|0|1|1|1|1|
|s2|0|1|1|1|1|1|



""")
    st.caption("Document - Term Frequency Matrix")

    st.markdown("""We can add nuance to the training data by opting to use term frequency inverse document frequency weighting (TFIDF) instead of term freqeuncy counts. TFIDF converts term frequencies into weights. Terms that are specific to particular sets of documents are upweighted in those documents. Terms that are ubiquitous across the entire corpus (e.g., pronouns, articles, prepositions) are downweighted. This techniques helps reduce the noise that innocuous textual data such as articles and prepositions, which will be highly frequent in all English texts regardless of genre.
                
A TFIDF-weighted document-term weight matrix of our toy corpus would resemble the following:
    
|sentence|a|another|is|sentence|test|this|
|--------|-|-------|--|--------|----|----|
|s1|.575|0|.409|.409|.409|.409|
|s2|0|.575|.409|.409|.409|.409|
    """)
    
    st.caption("Document - Term Weight Matrix")

    st.markdown("""
Notice that the term weights that differentiate s1 and s2 involve their disunion: the most distinguishing term weight for s1 is "a"--the word that s2 lacks. Conversely, "another" is the most distinguishing term for s2--the word that s1 lacks. These numerical differences allow the machine learning classifier to distinguish documents from each other.

Having converted the raw texts into TFIDF vectors, we can fit the training data and training labels to a logistic regression classifier. Each term-feature in the dataset will supply one coefficient to the logistic regression classifier per class (see Masaron and Boschetti, 2016).We can now query the model coefficients to derive the log-odds or make predictions on new data.                  

### How to Train a Logistic Regression Classifier?

- Format your .csv, .xsl, or .xlsx file thus:
    
    |code|text|
    |----|----|
    |D|this is a sentence}
    |D|the fire engine is red|
    |S|change your font|
    |E|Great job!|
- Upload your data as a .csv or .xlsx file.

:memo: `QC Utils` is mainly for quick demonstrations of qualitative coding metrics found in `sklearn`, `scipy`, `nltk`, and `statsmodels`. As such, its options to tune model parameters is limited. Users who need to train large volumes of coded textual data or require model calibration are encouraged to consult the example code and Python scientific libraries, which can return more robust results. 

### How to Evaluate Classification Results?
For classification tasks, we evaluate models based on the strength of their predictions. One standard metrics is accuracy (total correct decisions/total decisions). However, there are other metrics that can better assess the strength and quality of machine training. These include precision, recall, and f1-scores.

Consider the following table, which contains ground truth labels and classifier predictions:
    
|predicted|true|text|
|--|----|----|
|D|D|this is a sentence}
|D|D|the fire engine is red|
|D|S|change your font|
|E|E|Great job!|

   
    precision = true positives/ (true positive + false positives)
    
        For code D, classifier precision is 2/3 or 0.67.
        For code E, the classifier precision is 1/1 or 1.0
        For code S, the classifier precisions is 0/1 or 0.0.
        Mean precision is .56.

    recall = true positive/ total available positive
    
        For code D, classifier recall is 2/2 or 1.0.
        For code E, the classifier recall is 1/1 or 1.0.
        For code S, the classifier recall is 0/1 or 0.
        Mean recall is 0.67.

    f1-score = 2/precision + recall (the harmonic mean of precision and recall scores)
    
        For code D, the f1-score is 2/(1/.66) +(1/1) or 0.795.
        For code E, the f1-score is 2/(1/1) + (1/1) or 1.0
        For code S, the f1-socre is 2/(1/0) + (1/0) or 0.0.
        Mean f1-score is 0.60.
    
F1-scores balance classifier accuracy and classifier coverage, illustrating how reliably the classifier predicts a class and how relevant its predictions are. For example, a high precision score and a low recall score indicates that the classifier is accurate when assigning a specific label when it assigns it, but often misses relevant cases. Conseqeunently, we may be able to trust that a classifier's per class coding decision will be accurate, but cannot trust that the classifier has found a sufficient quantity of relevant samples. 

Generally, test sets--data held out of the training process--account for 20% of your total data (e.g., you train on 80% of your coded data).

f1-scores .80 and above testify to classifier precision and quality, although the sample size tested also plays an important role in determining classifier quality. High f1-scores achieved on a small test set may not be generalizable. Obviously, a test set of 5 classifier predictions would be insufficient in practice.

`QC Utils` also ranks the importance of textual features to class assignments. Coefficient importance is dervied by multiplying feature coefficients with the standard deviation of the feature's TFIDF weight (see scikit-learn developersB, n.d.). The logistic regression classifier  coefficients indicate the bearing of the log-odds of the training features. For binary classification, model coefficients will be positive and negative. The sign of the coefficient indicates the class that the feature coefficient tends towards. Negative coefficients describe the first class of the model; positive feature coefficients describe the second class. Beyond separating classes, the direction of feature coefficient signs is unimportant to model evaluation. What is important is the magnitude of the coefficient. 

In the non-binary case (three or more classes), feature coefficients can be difficult to interpret. For this reason, `QC Utils` allows you to drop class labels/segments and compare two model classes against each other.
""")
    uploaded_file = st.file_uploader("Upload your data as .csv or .xlsx file")
    st.caption("📝 This app does not retain user data.")
    if uploaded_file != None:
        train_lgr_text(uploaded_file)
    else:
        lgr_demo_text()
    st.markdown("""### Sample Python Code""")
    with st.expander('Sample Python Code'):
        st.code("""from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

#get 20 Newsgroups Data
news = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'comp.graphics'],remove=('headers','footers','quotes'))

#get labels
atheism = [news.data[i] for i in range(len(news)) if news.target[i] == 0]
graphics = [news.data[i] for i in range(len(news)) if news.target[i] == 1]
          
#instantiate LogisticRegression class
lgr_d = LogisticRegression()

#split data into training and testing sets
X_train, x_test, y_train, y_test = train_test_split(news.data[:5000],news.target[:5000])

#instantiate TfidfVectorizer class
tfidf = TfidfVectorizer()

#vectorize training data
train_vecs = tfidf.fit_transform(X_train)

#fit LogisticRegression Classifier
lgr_d.fit(train_vecs,y_train)

#make predictions on test data
predictions = lgr_d.predict(tfidf.transform(x_test))

#print classification results of predictions
print(classification_report(y_test, predictions))   

                """)
    
def train_lgr_text(uploaded_file):
    if str(uploaded_file.name).endswith('csv'):
        
        to_drop = st.text_input('Enter the target values to drop. Separate a list of target values with a space.',)
        if to_drop != None:
            df = pd.read_csv(uploaded_file,encoding_errors='ignore')
            df = pd.DataFrame(df.iloc[i] for i in range(len(df)) if df.code.tolist()[i] not in to_drop)
            st.subheader('Results')
            texts = df['text'].tolist()
            labels = df['code'].tolist()
            X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.20, random_state=42)
            lgr = LogisticRegression()
            tfidf = TfidfVectorizer(stop_words='english',min_df=5)
            tvecs = tfidf.fit_transform(X_train)
            test_vecs = tfidf.transform(X_test)
            lgr.fit(tvecs, y_train)
            
            e0, e1, e2 = st.columns(3)
            e0.metric('Total Samples',len(X_train + X_test)) 
            e1.metric('Dataset Word Count',len(' '.join(X_train + X_test).split()))
            e2.metric('Dataset Unique Words Count',len(list(set(' '.join(X_train + X_test).lower().split()))))
            
            c0, c1, c2 = st.columns(3)
            c0.metric('# of Training Samples',len(X_train))
            
            c1.metric('Training Set Word Count',len(' '.join(X_train).split()))
            c2.metric('Training Set Unique Words Count',len(list(set(' '.join(X_train).lower().split()))))
            
            t0, t1, t2 = st.columns(3)
            t0.metric('Total Test Samples',len(X_test)) 
            t1.metric('Test Set Word Count',len(' '.join(X_test).split()))
            t2.metric('Test Set Unique Words Count',len(list(set(' '.join(X_test).lower().split()))))
            
            st.markdown('### Training Data Code Distribution')
            fig = plt.figure(figsize=(3,1))
            ax = plt.axes()
            ax.bar(Counter(y_train).keys(), Counter(y_train).values())
            st.pyplot(fig)
            predictions = lgr.predict(test_vecs)
            
            
            
            tc0,tc1 = st.columns(2)
            
    
    
            tc0.markdown("""### Classifier Parameters""")
            
            tc0.write(lgr.get_params())    
            
            tc1.markdown("### Classification Report")
            
            predictions = lgr.predict(tfidf.transform(X_test))
            
            tc1.text('Accuracy: '+ str(accuracy_score(y_test, predictions)))
            tc1.text(classification_report(y_test, predictions))
            
            feats = list(zip(lgr.coef_[0],tfidf.get_feature_names()))
            std_coef = pd.DataFrame(tvecs.todense()).std(axis=0).tolist()
            ff = pd.DataFrame({'feature':[y for (x,y) in feats],'coef':[feats[i][0] * std_coef[i] for i in range(len(feats))]})
            
            #ff['coef'] = pd.to_numeric(ff['coef'])
            tc2, tc3 =st.columns(2)
            
            tc2.markdown("""#### Decision Function Coeffecients""")
            
            fff = ff.sort_values('coef',ascending=True)
            
            
            tc2.write(fff)        
            
            tc2.caption(""":memo: Coefficients multiplied by standard deviation of the original TFIDF scores to derive their importance. The magnitude and sign (- or +) of the coefficient determine feature contribution to class assignment. Negative coefficients inform the -1 class (`class_0`); positive coefficients inform the positive class (`class_1`). The size of the coefficient attests to its influence on the model's decision-making.""")
            
            tc3.markdown("""#### Plot of Coefficient Importance""")
            fig = px.scatter(fff,x=list(range(len(fff))),y='coef',hover_data=['feature'])
            tc3.plotly_chart(fig)
            tc3.caption("Terms with coefficients closer to -1 contribute more to `class_0`. Terms with coefficients closer to one are more likely to contribute to `class_1`. Terms closer to 0 could belong to either class.")
            submission = st.text_input('Classify a text with your trained model.',value='')
    
            if submission != '':
                submission_vec = tfidf.transform([submission])
                st.metric('Prediction',lgr.predict(submission_vec)[0])
        else:
            df = pd.read_csv(uploaded_file,encoding_errors='ignore')
            st.subheader('Results')
            texts = df['text'].tolist()
            labels = df['code'].tolist()
            X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.33, random_state=42)
            lgr = LogisticRegression()
            tfidf = TfidfVectorizer(stop_words='english',min_df=5)
            tvecs = tfidf.fit_transform(X_train)
            test_vecs = tfidf.transform(X_test)
            lgr.fit(tvecs, y_train)
            
            e0, e1, e2 = st.columns(3)
            e0.metric('Total Samples',len(X_train + X_test)) 
            e1.metric('Dataset Word Count',len(' '.join(X_train + X_test).split()))
            e2.metric('Dataset Unique Words Count',len(list(set(' '.join(X_train + X_test).lower().split()))))
            
            c0, c1, c2 = st.columns(3)
            c0.metric('# of Training Samples',len(X_train))
            
            c1.metric('Training Set Word Count',len(' '.join(X_train).split()))
            c2.metric('Training Set Unique Words Count',len(list(set(' '.join(X_train).lower().split()))))
            
            t0, t1, t2 = st.columns(3)
            t0.metric('Total Test Samples',len(X_test)) 
            t1.metric('Test Set Word Count',len(' '.join(X_test).split()))
            t2.metric('Test Set Unique Words Count',len(list(set(' '.join(X_test).lower().split()))))
            
            st.markdown('### Training Data Code Distribution')
            fig = plt.figure(figsize=(3,1))
            ax = plt.axes()
            ax.bar(Counter(y_train).keys(), Counter(y_train).values())
            st.pyplot(fig)
            predictions = lgr.predict(test_vecs)
            
            
            
            tc0,tc1 = st.columns(2)
            


            tc0.markdown("""### Classifier Parameters""")
            
            tc0.write(lgr.get_params())    
            
            tc1.markdown("### Classification Report")
            
            predictions = lgr.predict(tfidf.transform(X_test))
            
            tc1.text('Accuracy: '+ str(accuracy_score(y_test, predictions)))
            tc1.text(classification_report(y_test, predictions))
            
            feats = list(zip(lgr.coef_[0],tfidf.get_feature_names()))
            std_coef = pd.DataFrame(tvecs.todense()).std(axis=0).tolist()
            ff = pd.DataFrame({'feature':[y for (x,y) in feats],'coef':[feats[i][0] * std_coef[i] for i in range(len(feats))]})
            
            #ff['coef'] = pd.to_numeric(ff['coef'])
            tc2, tc3 =st.columns(2)
            
            tc2.markdown("""#### Decision Function Coeffecients""")
            
            fff = ff.sort_values('coef',ascending=False)
            tc2.write(fff)        
            
            tc2.caption(""":memo: Coefficients multiplied by standard deviation of the original TFIDF scores to derive their importance. The magnitude and sign (- or +) of the coefficient determine feature contribution to class assignment. Negative coefficients inform the -1 class (`class_0`); positive coefficients inform the positive class (`class_1`). The size of the coefficient attests to its influence on the model's decision-making.""")
            
            tc3.markdown("""#### Plot of Coefficient Importance""")
            fig = px.scatter(fff,x=fff.index,y='coef',hover_data=['feature'])
            tc3.plotly_chart(fig)
            tc3.caption("Terms with coefficients closer to -1 contribute more to `class_0`. Terms with coefficients closer to one are more likely to contribute to `class_1`. Terms closer to 0 could belong to either class.")
            submission = st.text_input('Classify a text with your trained model.',value='')

            if submission != '':
                submission_vec = tfidf.transform([submission])
                st.metric('Prediction',lgr.predict(submission_vec)[0])
        

        download_model(lgr)
    elif str(uploaded_file.name).endswith('xls'):
        to_drop = st.text_input('Enter the target values to drop. Separate a list of target values with a space.',)
        if to_drop != None:
            df = pd.read_excel(uploaded_file)
            df = pd.DataFrame(df.iloc[i] for i in range(len(df)) if df.code.tolist()[i] not in to_drop)
            st.subheader('Results')
            texts = df['text'].tolist()
            labels = df['code'].tolist()
            X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.20, random_state=42)
            lgr = LogisticRegression()
            tfidf = TfidfVectorizer(stop_words='english',min_df=5)
            tvecs = tfidf.fit_transform(X_train)
            test_vecs = tfidf.transform(X_test)
            lgr.fit(tvecs, y_train)
            
            e0, e1, e2 = st.columns(3)
            e0.metric('Total Samples',len(X_train + X_test)) 
            e1.metric('Dataset Word Count',len(' '.join(X_train + X_test).split()))
            e2.metric('Dataset Unique Words Count',len(list(set(' '.join(X_train + X_test).lower().split()))))
            
            c0, c1, c2 = st.columns(3)
            c0.metric('# of Training Samples',len(X_train))
            
            c1.metric('Training Set Word Count',len(' '.join(X_train).split()))
            c2.metric('Training Set Unique Words Count',len(list(set(' '.join(X_train).lower().split()))))
            
            t0, t1, t2 = st.columns(3)
            t0.metric('Total Test Samples',len(X_test)) 
            t1.metric('Test Set Word Count',len(' '.join(X_test).split()))
            t2.metric('Test Set Unique Words Count',len(list(set(' '.join(X_test).lower().split()))))
            
            st.markdown('### Training Data Code Distribution')
            fig = plt.figure(figsize=(3,1))
            ax = plt.axes()
            ax.bar(Counter(y_train).keys(), Counter(y_train).values())
            st.pyplot(fig)
            predictions = lgr.predict(test_vecs)
            st.markdown('### Classification Report')
            
            
            
            tc0,tc1 = st.columns(2)
            
    
    
            tc0.markdown("""### Classifier Parameters""")
            
            tc0.write(lgr.get_params())    
            
            tc1.markdown("### Classification Report")
            
            predictions = lgr.predict(tfidf.transform(X_test))
            
            tc1.text('Accuracy: '+ str(accuracy_score(y_test, predictions)))
            tc1.text(classification_report(y_test, predictions))
            
            feats = list(zip(lgr.coef_[0],tfidf.get_feature_names()))
            std_coef = pd.DataFrame(tvecs.todense()).std(axis=0).tolist()
            ff = pd.DataFrame({'feature':[y for (x,y) in feats],'coef':[feats[i][0] * std_coef[i] for i in range(len(feats))]})
            
            #ff['coef'] = pd.to_numeric(ff['coef'])
            tc2, tc3 =st.columns(2)
            
            tc2.markdown("""#### Decision Function Coeffecients""")
            
            fff = ff.sort_values('coef',ascending=True)
            
            
            tc2.write(fff)        
            
            tc2.caption(""":memo: Coefficients multiplied by standard deviation of the original TFIDF scores to derive their importance. The magnitude and sign (- or +) of the coefficient determine feature contribution to class assignment. Negative coefficients inform the -1 class (`class_0`); positive coefficients inform the positive class (`class_1`). The size of the coefficient attests to its influence on the model's decision-making.""")
            
            tc3.markdown("""#### Plot of Coefficient Importance""")
            fig = px.scatter(fff,x=list(range(len(fff))),y='coef',hover_data=['feature'])
            tc3.plotly_chart(fig)
            tc3.caption("Terms with coefficients closer to -1 contribute more to `class_0`. Terms with coefficients closer to one are more likely to contribute to `class_1`. Terms closer to 0 could belong to either class.")
            submission = st.text_input('Classify a text with your trained model.',value='')
    
            if submission != '':
                submission_vec = tfidf.transform([submission])
                st.metric('Prediction',lgr.predict(submission_vec)[0])

        download_model(lgr)

def encode_df(dataframe):
    
    train_enc = OneHotEncoder()
    target_enc = LabelEncoder()
    
    
    dff = dataframe.drop(columns='target')
    dff.sample(frac=1).reset_index(drop=True)
    
    # train = pd.DataFrame([Counter(dff.iloc[i]) for i in range(len(dff))])
    # train.fillna(0,inplace=True)
    train_data = train_enc.fit_transform(dataframe[[col for col in dataframe.columns if col != 'target']]).toarray()
    #target_endog = dataframe['target']
    
    targets = dataframe['target']
    target_endog = target_enc.fit_transform(targets).astype(float)
      
    return train_data, target_endog, train_enc

def train_lgr_table(uploaded_file,encoder_option):
    
    
    if str(uploaded_file.name).endswith('csv'):
        df = pd.read_csv(uploaded_file)
        st.subheader('Results')
        
        st.markdown("""#### Uploaded Data""")
        
        st.dataframe(df)
        to_drop = st.text_input('Enter the target values to drop. Separate a list of target values with a space.',)
        st.markdown(sorted(list(set(df.target))))

        
        if encoder_option == True:
            
            
            if to_drop != None:
                td = to_drop.split()
                
                
                if len(td) > len(list(set(df.target.tolist())))-1:
                    st.error("Too many target values dropped.")
                    st.stop()
                else: 
                    tf = pd.DataFrame([df.iloc[i] for i in range(len(df)) if df.iloc[i].target not in td])
                    #cats = flatten_list([list(train_enc.categories_[i]) for i in range(len(train_enc.categories_))])
                    train, target_endog,train_enc = encode_df(tf)

                    X_train, X_test, y_train, y_test = train_test_split(train, target_endog, test_size=0.33, random_state=42)
                    
                    
                    st.markdown("""Target Values""")
                    #train_ = [X_train.tolist()[i] for i in range(len(X_train)) if y_train[i] not in td]
                    train_data = pd.DataFrame(sm.add_constant(X_train,prepend=False,has_constant='add'))
            
                    logit_mod = sm.MNLogit(y_train,train_data, method='lbfgs')
                    logit_res = logit_mod.fit()
                    st.write(logit_res.summary())
                    
                    lo, pr = st.columns(2)
                    lo.markdown("""#### Odds Ratio""")
                    lo.write(np.exp(logit_res.params))
                    # st.markdown("""#### Odds Ratio""")
                    
                        
                    lf = pd.DataFrame(np.exp(logit_res.params))
                    lff = lf.applymap(lambda x: x/(1+x))   
                    pr.markdown("""#### Probability""")  
                    pr.dataframe(lff)
                    
                    #st.markdown("""#### Odds Ratio""")
                    #lf = pd.DataFrame(np.exp(logit_res.params[:-1]))
                    
                    
                    
                    #lf = pd.DataFrame(np.exp(logit_res.params))
                   
                    #st.write(train_enc.categories_)
                   # st.write(lf)
                    
                    
                    st.markdown("""#### Marginal Effects
        
Marginal effects indicate the influence of an independent variable on the dependent or response variable. 
                                """)
                    st.write(logit_res.get_margeff().summary())
                         
                    x_test = sm.add_constant(X_test, prepend=False, has_constant='add')
                    predictions = logit_res.predict(x_test)
                    p = [np.argmax(predictions[i]) for i in range(len(predictions))]
                    st.text(classification_report(y_test, p))
                    
            else:
                train = df[[col for col in df.columns if col != 'target']]
                target_endog = df['target']
                    
                X_train, X_test, y_train, y_test = train_test_split(train, target_endog, test_size=0.33, random_state=42)
                train_data = sm.add_constant(X_train, prepend=False, has_constant='add')
        
                logit_mod = sm.MNLogit(y_train,train_data, method='lbfgs')
                logit_res = logit_mod.fit()
                st.write(logit_res.summary())
                
                lo, pr = st.columns(2)
                lo.markdown("""#### Odds Ratio""")
                lo.write(np.exp(logit_res.params))
                # st.markdown("""#### Odds Ratio""")
                
                    
                lf = pd.DataFrame(np.exp(logit_res.params))
                lff = lf.applymap(lambda x: x/(1+x))   
                pr.markdown("""#### Probability""")  
                pr.dataframe(lff)
                    
                # st.markdown("""#### Odds Ratio""")
                    
                    
                # lf = pd.DataFrame(np.exp(logit_res.params)) 
                # st.write(lf)
                    
                    
                st.markdown("""#### Marginal Effects
       
Marginal effects indicate the influence of an independent variable on the dependent or response variable. 
                                """)
                st.write(logit_res.get_margeff().summary())
                    
                x_test = sm.add_constant(X_test, prepend=False, has_constant='add')
                predictions = logit_res.predict(x_test)
                p = [np.argmax(predictions.iloc[i]) for i in range(len(predictions))]
                st.text(classification_report(y_test, p))
        else:
            train = df[[col for col in df.columns if col != 'target']]
            target_endog = df['target']
                
            X_train, X_test, y_train, y_test = train_test_split(train, target_endog, test_size=0.33, random_state=42)
            train_data = sm.add_constant(X_train, prepend=False, has_constant='add')
    
            logit_mod = sm.MNLogit(y_train,train_data, method='lbfgs')
            logit_res = logit_mod.fit()
            st.write(logit_res.summary())
            
            lo, pr = st.columns(2)
            lo.markdown("""#### Odds Ratio""")
            lo.write(np.exp(logit_res.params))
            # st.markdown("""#### Odds Ratio""")
            
                
            lf = pd.DataFrame(np.exp(logit_res.params))
            lff = lf.applymap(lambda x: x/(1+x))   
            pr.markdown("""#### Probability""")  
            pr.dataframe(lff)
                
          
                
            st.markdown("""#### Marginal Effects
    
Marginal effects indicate the influence of an independent variable on the dependent or response variable. 
                            """)
            st.write(logit_res.get_margeff().summary())
            
            x_test = sm.add_constant(X_test, prepend=False, has_constant='add')
            predictions = logit_res.predict(x_test)
            p = [np.argmax(predictions.iloc[i]) for i in range(len(predictions))]
            st.text(classification_report(y_test, p))
        
            
            #st.write()
            #st.error('Looks like your data is categorical. Trying selecting "One-Hot Encode." ')
        

    #     download_model(lgr)
    elif str(uploaded_file.name).endswith('xls'):
        df = pd.read_excel(uploaded_file)
        st.subheader('Results')
        
        st.markdown("""#### Uploaded Data""")
        
        st.dataframe(df)
        to_drop = st.text_input('Enter the target values to drop. Separate a list of target values with a space.',)
        st.markdown(sorted(list(set(df.target))))

        
        if encoder_option == True:
            
            
            if to_drop != None:
                td = to_drop.split()
                
                
                if len(td) > len(list(set(df.target.tolist())))-1:
                    st.error("Too many target values dropped.")
                    st.stop()
                else: 
                    tf = pd.DataFrame([df.iloc[i] for i in range(len(df)) if df.iloc[i].target not in td])
                    #cats = flatten_list([list(train_enc.categories_[i]) for i in range(len(train_enc.categories_))])
                    train, target_endog,train_enc = encode_df(tf)

                    X_train, X_test, y_train, y_test = train_test_split(train, target_endog, test_size=0.33, random_state=42)
                    
                    
                    st.markdown("""Target Values""")
                    #train_ = [X_train.tolist()[i] for i in range(len(X_train)) if y_train[i] not in td]
                    train_data = pd.DataFrame(sm.add_constant(X_train,prepend=False,has_constant='add'))
            
                    logit_mod = sm.MNLogit(y_train,train_data, method='lbfgs')
                    logit_res = logit_mod.fit()
                    st.write(logit_res.summary())
                    
                    lo, pr = st.columns(2)
                    lo.markdown("""#### Odds Ratio""")
                    lo.write(np.exp(logit_res.params))
                    # st.markdown("""#### Odds Ratio""")
                    
                        
                    lf = pd.DataFrame(np.exp(logit_res.params))
                    lff = lf.applymap(lambda x: x/(1+x))   
                    pr.markdown("""#### Probability""")  
                    pr.dataframe(lff)
                    #st.markdown("""#### Odds Ratio""")
                    #lf = pd.DataFrame(np.exp(logit_res.params[:-1]))
                    
                    
                    
                    # lf = pd.DataFrame(np.exp(logit_res.params))
                   
                    # #st.write(train_enc.categories_)
                    # st.write(lf)
                    
                    
                    st.markdown("""#### Marginal Effects
Marginal effects indicate the influence of an independent variable on the dependent or response variable. 
                                """)
                    st.write(logit_res.get_margeff().summary())
                       
                    x_test = sm.add_constant(X_test, prepend=False, has_constant='add')
                    predictions = logit_res.predict(x_test)
                    p = [np.argmax(predictions[i]) for i in range(len(predictions))]
                    st.text(classification_report(y_test, p))
                    
            else:
                train = df[[col for col in df.columns if col != 'target']]
                target_endog = df['target']
                    
                X_train, X_test, y_train, y_test = train_test_split(train, target_endog, test_size=0.33, random_state=42)
                train_data = sm.add_constant(X_train, prepend=False, has_constant='add')
        
                logit_mod = sm.MNLogit(y_train,train_data, method='lbfgs')
                logit_res = logit_mod.fit()
                st.write(logit_res.summary())
                
                lo, pr = st.columns(2)
                lo.markdown("""#### Odds Ratio""")
                lo.write(np.exp(logit_res.params))
                # st.markdown("""#### Odds Ratio""")
                
                    
                lf = pd.DataFrame(np.exp(logit_res.params))
                lff = lf.applymap(lambda x: x/(1+x))   
                pr.markdown("""#### Probability""")  
                pr.dataframe(lff)
                    
                # st.markdown("""#### Odds Ratio""")
                    
                    
                # lf = pd.DataFrame(np.exp(logit_res.params)) 
                # st.write(lf)
                    
                    
                st.markdown("""#### Marginal Effects
Marginal effects indicate the influence of an independent variable on the dependent or response variable. 
                                """)
                st.write(logit_res.get_margeff().summary())
                    
                x_test = sm.add_constant(X_test, prepend=False, has_constant='add')
                predictions = logit_res.predict(x_test)
                p = [np.argmax(predictions.iloc[i]) for i in range(len(predictions))]
                st.text(classification_report(y_test, p))
        else:
            train = df[[col for col in df.columns if col != 'target']]
            target_endog = df['target']
                
            X_train, X_test, y_train, y_test = train_test_split(train, target_endog, test_size=0.33, random_state=42)
            train_data = sm.add_constant(X_train, prepend=False, has_constant='add')
    
            logit_mod = sm.MNLogit(y_train,train_data, method='lbfgs')
            logit_res = logit_mod.fit()
            st.write(logit_res.summary())
            
            lo, pr = st.columns(2)
            lo.markdown("""#### Odds Ratio""")
            lo.write(np.exp(logit_res.params))
            # st.markdown("""#### Odds Ratio""")
            
                
            lf = pd.DataFrame(np.exp(logit_res.params))
            lff = lf.applymap(lambda x: x/(1+x))   
            pr.markdown("""#### Probability""")  
            pr.dataframe(lff)
                
            # st.markdown("""#### Odds Ratio""")
                
                
            # lf = pd.DataFrame(np.exp(logit_res.params)) 
            # st.write(lf)
                
                
            st.markdown("""#### Marginal Effects
    
Marginal effects indicate the influence of an independent variable on the dependent or response variable. 
                            """)
            st.write(logit_res.get_margeff().summary())
              
            x_test = sm.add_constant(X_test, prepend=False, has_constant='add')
            predictions = logit_res.predict(x_test)
            p = [np.argmax(predictions.iloc[i]) for i in range(len(predictions))]
            st.text(classification_report(y_test, p))
        

    #     download_model(lgr)
        
def train_lnr_table(uploaded_file):

    housing = fetch_california_housing(as_frame=True)

    if str(uploaded_file.name).endswith('csv'):
        df = pd.read_csv(uploaded_file,encoding_errors='ignore')
        train_data = df[[col for col in df.columns if col != 'target']]
        targets = df['target'].tolist()
        
        
        
        
        X_train, X_test, y_train, y_test = train_test_split(train_data, targets, test_size=0.33, random_state=42)
        
       
        #lnr.fit(X_train, y_train)
        _exog = sm.add_constant(X_train, prepend=False, has_constant='add')
        mod = sm.OLS(y_train, _exog)
        res = mod.fit()

        X_test_exog = sm.add_constant(X_test,prepend=False,has_constant='add')
        predictions = res.predict(X_test_exog)
        
        st.markdown("""### Results""")

        fig, ax = plt.subplots()
        ax.plot(range(len(X_test)),y_test,color='b')
        ax.plot(range(len(X_test)),predictions,color='r')
        ax.legend(['Observed','Predicted'])
        plt.title('Linear Regression Plot')
        st.pyplot(fig)
        st.caption("Linear regression plot")
        
        st.markdown('#### Summary Statistics (provided by `statsmodels` API)')
        
        st.write(res.summary())

        # lnr = LinearRegression(fit_intercept=True)
        # predictions = lnr.predict(X_test)
        # fig, ax = plt.subplots()
        # ax.scatter(range(len(X_test)),y_test,color='b')
        # ax.plot(range(len(X_test)),predictions,color='r')
        # st.pyplot(fig)
        
        # st.write("Slope: ",lnr.coef_[0])
        

        
        # st.write("Intercept: ",lnr.intercept_)
        # st.write('Linear Regressor Coefficients')
        # lf = pd.DataFrame({'features':housing.data.columns, "coefficients":lnr.coef_})
        # st.dataframe(lf)
        
        
        # download_model(lnr)
    elif str(uploaded_file.name).endswith('xlsx'):
        df = pd.read_csv(uploaded_file,encoding_errors='ignore')
        st.subheader('Results')
        train_data = df[[col for col in df.columns if col != 'target']] 
        targets = df['target'].tolist()
        
        lnr = LinearRegression(fit_intercept=True)
        predictions = lnr.predict(X_test)
        fig, ax = plt.subplots()
        ax.scatter(range(len(X_test)),y_test,color='b')
        ax.plot(range(len(X_test)),predictions,color='r')
        st.pyplot(fig)
        
        st.write("Slope: ",lnr.coef_[0])
        

        
        st.write("Intercept: ",lnr.intercept_)
        st.write('Linear Regressor Coefficients')
        lf = pd.DataFrame({'features':housing.data.columns, "coefficients":lnr.coef_})
        st.dataframe(lf)
        
        
        download_model(lnr)
    else:
        lnr_demo()
    
def log_odds_to_proba(odds):
    return odds/(1+odds)


def lgr_demo_tabular():
    st.subheader("Multinomial Logistic Regression Demo of `scikit-learn's` Wine Dataset")
    wine = load_wine(as_frame=True)
    with st.expander("Wine Dataset Description (from sklearn API)"):
        st.write(wine.DESCR)
        st.markdown('Sample Wine Data')
        st.write(wine.data.head())
    
    to_drop = st.text_input('Enter the target values to drop. Separate a list of target values with a space.',)
    st.markdown("""Target Values""")
    st.markdown(set(wine.target))
    
    
    if to_drop != None:
        td = to_drop.split()
        if len(td) > len(list(set(wine.target)))-1:
            st.error("Too many target values dropped.")
            st.stop()
        else:      
            wf = [wine.data.iloc[i] for i in range(len(wine.data)) if str(wine.target[i]) not in td]
            wine_data = sm.add_constant(pd.DataFrame(wf),prepend=False, has_constant='add')
            y = [str(t) for t in wine.target if str(t) not in td]
                
            
            X_train, x_test, y_train, y_test = train_test_split(wine_data, y,test_size=.2)    
            logit_mod = sm.MNLogit(y_train, X_train)
            logit_res = logit_mod.fit(method='lbfgs')
                
            st.markdown('#### Summary Statistics (provided by the statsmodels `API`)')
            
                #print model summary statistics
            
            st.write(logit_res.summary())
            
            ora, lo, pr = st.columns(3)
            ora.markdown("""#### Odds Ratio""")
            ora.write(np.exp(logit_res.params))
            # st.markdown("""#### Odds Ratio""")
            
            
                
            lf = pd.DataFrame(np.exp(logit_res.params))
            lff = lf.applymap(lambda x: x/(1+x))   
            
            lo.markdown("""#### log Odds""")
            log_odds = lf.applymap(lambda x: np.log(x))
            lo.dataframe(log_odds)
            pr.markdown("""#### Probability""")  
            pr.dataframe(lff)
                
            st.markdown("""#### Marginal Effects
                        
Marginal effects indicate the influence of an independent variable on the dependent or response variable. 
                            """)
            st.write(logit_res.get_margeff().summary())
                
            st.markdown("""#### Classification Report
                            """)
            predictions = logit_res.predict(x_test)
            p = [str(np.argmax(predictions.iloc[i])) for i in range(len(predictions))]
            st.text(classification_report(y_test, p))
    else:
            #wf = [wine.data.iloc[i] for i in range(len(wine.data)) if str(wine.target[i]) not in td]
        wine_data = sm.add_constant(wf,prepend=False, has_constant='add')
        y = wine.target
                
            
        X_train, x_test, y_train, y_test = train_test_split(wine_data, y,test_size=.2)    
        logit_mod = sm.MNLogit(y_train, X_train)
        logit_res = logit_mod.fit(method='lbfgs')
                
        st.markdown('#### Summary Statistics (provided by the statsmodels `API`)')
            
                #print model summary statistics
        st.write(logit_res.summary())
        
        lo, pr = st.columns(2)
        lo.markdown("""#### Odds Ratio""")
        lo.write(np.exp(logit_res.params))
        # st.markdown("""#### Odds Ratio""")
        
            
        lf = pd.DataFrame(np.exp(logit_res.params))
        lff = lf.applymap(lambda x: x/(1+x))   
        pr.markdown("""#### Probability""")  
        pr.dataframe(lff)
        
        # st.markdown("""#### Odds Ratio""")
                
                
        # lff = pd.DataFrame(np.exp(logit_res.params))
                
                
        # st.dataframe(lff)
                
        st.markdown("""#### Marginal Effects
            
Marginal effects indicate the influence of an independent variable on the dependent or response variable. 
                            """)
        st.write(logit_res.get_margeff().summary())
                
        st.markdown("""#### Classification Report
                            """)
        predictions = logit_res.predict(x_test)
        p = [str(np.argmax(predictions[i])) for i in range(len(predictions))]
        st.text(classification_report(y_test, p))
        
def lgr_demo_text():
    st.subheader("Logistic Regression Classification Results of `scikit-learn's` 20 Newsgroups Dataset")
    
    st.write("""
    This demo logistic regression classifier is trained on the 'alt.atheism' and 'comp.graphics' posts of scikit-learn's `20_newsgroups` dataset. 
    
    The classifier determines whether a post most likely comes from `alt.atheism` (0) or `comp.graphics` (1).
    """)
    news = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'comp.graphics'],remove=('headers','footers','quotes'))
    
    atheism = [news.data[i] for i in range(len(news.data)) if news.target[i] == 0]
    
    a_target = [0] * len(atheism)
    graphics = [news.data[i] for i in range(len(news.data)) if news.target[i] == 1]
    g_target = [1] * len(graphics)
    
    

    with st.expander("20 Newsgroups Dataset Description (from sklearn API)"):
         st.write(fetch_20newsgroups().DESCR)
         st.markdown("### Sample `alt.atheism` post")
         st.write(atheism[0])
         st.markdown("### Sample `comp.graphics` post")
         st.write(graphics[0])
         
         
         
    lgr_d = LogisticRegression(multi_class='multinomial')
    X_train, x_test, y_train, y_test = train_test_split(atheism + graphics,a_target + g_target)
    
    tfidf = TfidfVectorizer(stop_words='english',min_df=3)
    train_vecs = tfidf.fit_transform(X_train)
    lgr_d.fit(train_vecs,y_train)
    
    c0, c1, c2 = st.columns(3)
    c0.metric('# of Training Samples',len(train_vecs.todense()))
    c1.metric('Total Word Count',len(' '.join(atheism + graphics).split()))
    c2.metric('# Unique Words',len(list(set(' '.join(atheism + graphics).lower().split()))))
    st.markdown('### Training Data Code Distribution')
    fig, ax = plt.subplots()
    
    ax.bar([str(x) for x in Counter(y_train).keys()], Counter(y_train).values())
    st.pyplot(fig)
    
    tc0,tc1 = st.columns(2)
    


    tc0.markdown("""### Classifier Parameters""")
    
    tc0.write(lgr_d.get_params())    
    
    tc1.markdown("### Classification Report")
    
    predictions = lgr_d.predict(tfidf.transform(x_test))
    
    tc1.text('Accuracy: '+ str(accuracy_score(y_test, predictions)))
    tc1.text(classification_report(y_test, predictions))
    
    feats = list(zip(lgr_d.coef_[0],tfidf.get_feature_names()))
    #st.write(pd.DataFrame(lgr_d.coef_[0]).std)
    
    std_coef = pd.DataFrame(train_vecs.todense()).std(axis=0).tolist()
    ff = pd.DataFrame({'feature':[y for (x,y) in feats],'coef':[feats[i][0] * std_coef[i] for i in range(len(feats))]})
    
    fff = ff.sort_values('coef',ascending=False)
    tc2, tc3 =st.columns(2)
    
    tc2.markdown("""#### Decision Function Coeffecients""")
    tc2.write(fff)
    tc2.caption(""":memo: Coefficients multiplied by standard deviation of the original TFIDF scores scores to derive their importance (see scikit-learn developersB,n.d.). The magnitude and sign (- or +) of the coefficient determine feature contribution to class assignment. Negative coefficients inform the -1 class (`class_0`); positive coefficients inform the positive class (`class_1`). The size of the coefficient attests to its influence on the model's decision-making.""")
    
    tc3.markdown("""#### Plot of Coefficient Importance""")
    fig = px.scatter(fff,x=ff.index,y='coef',hover_data=['feature'])
    tc3.plotly_chart(fig)
    tc3.caption("Terms with coefficients closer to -1 contribute more to `class_0`. Terms with coefficients closer to one are more likely to contribute to `class_1`. Terms closer to 0 could belong to either class.")
def download_model(model):
    """
    Code quoted Streamlit. (May 2020). How to download a trained model. Retrieved from https://discuss.streamlit.io/t/how-to-download-a-trained-model/2976.
    """
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="clf.pkl">Download Trained Logistic Regression Classifier as pickle.</a>'
    st.markdown(href, unsafe_allow_html=True)
       
def main():
    
    st.sidebar.markdown("""<h1 style="font-size:4em; font-weight:strong">QC Utils</h1><h3>Metrics for Qualitative and Mixed Methods Coding Research</h3>""",unsafe_allow_html=True)
    
        
    options = st.sidebar.selectbox('Navigation',('About',"Cohen's Kappa","Krippendorff's Alpha","chi2 Goodness of Fit", "chi2 Homogeneity","Linear Regression", 'Logistic Regression Classification Text', 'Logistic Regression Classification Tabular'))
    
    if options == 'About':
        about()
    elif options == "Cohen's Kappa":
        data_options = st.sidebar.selectbox('How would you like to submit your data?',("Copy and Paste","Upload .csv or .xlsx"))
        if data_options == "Copy and Paste":
        
            kappa()
        else:
            kappa_file_upload()
    elif options == "Krippendorff's Alpha":
        data_options = st.sidebar.selectbox('How would you like to submit your data?',("Copy and Paste","Upload .csv or .xlsx"))
        if data_options == "Copy and Paste":
        
            k_alpha()
        else:
            k_alpha_file_upload()
    elif options == 'chi2 Homogeneity':
        data_options = st.sidebar.selectbox('How would you like to submit your data?',("Copy and Paste","Upload .csv or .xlsx"))
        if data_options == "Copy and Paste":
            chi()
        else:
            chi_file_upload()
    elif options == 'chi2 Goodness of Fit':
        data_options = st.sidebar.selectbox('How would you like to submit your data?',("Copy and Paste","Upload .csv or .xlsx"))
        if data_options == "Copy and Paste":
            chi_goodness()
        else:
            chi_goodness_file_upload()
    elif options == 'Linear Regression':
        data_options = st.sidebar.selectbox('How would you like to submit your data?', ("Upload .csv, .xls, or .xlsx",))
        lnr_fit()
    elif options == 'Logistic Regression Classification Text':
        data_options = st.sidebar.selectbox('How would you like to submit your data?', ("Upload .csv, .xls, or .xlsx",))
        lgr_classify_text()
    elif options == 'Logistic Regression Classification Tabular':
        data_options = st.sidebar.selectbox('How would you like to submit your data?',("Upload .csv, .xls, or .xlsx",))

        lgr_classify_table()
    st.sidebar.markdown("""
If you use this app for research, please cite as:

Omizo, R. (2022). QC Utils. [Software]. Retrieved from https://share.streamlit.io/rmomizo/zsclf/main/kappa_st.py.

© Ryan Omizo 2022
""")


def about():
    st.title("About")
    
    st.markdown("### Overview")
    st.write("**QC Utils** demos key metrics from Cheryl Geisler and Jason Swarts' (2019) [_Coding Streams of Language: Techniques for the Systematic Coding of Text, Talk, and Other Verbal Data_](https://wac.colostate.edu/books/practice/codingstreams/).")
    st.write("**QC Utils** serves as wrapper for the Python statistical processing libraires `scikit-learn` (Pedrogosa et al., 2013), `The Natural Language Toolkit (NLTK)`,`statsmodels`, and `scipy` (Virtanen et al., 2020) built and hosted by [Streamlit](https://streamlit.io/).")
    
    st.markdown("""
    With `QC Utils`, you can run:
    - Cohen's Kappa inter-rater agreement tests for two coders
    - Krippendoff's Alpha inter-rater agreement tests for two coders
    - chi2 goodness of fit test for single samples
    - chi2 test of homogeneity for two samples
    - linear regression (tabular data)
    - logistic regression classification (text and tabular data) 
    """)
    
    st.markdown("### Introduction")
    st.markdown("""
Cheryl Geisler and Jason Swarts' (2019) guide to mixed-methods coding research, [_Coding Streams of Langauge: Techniques for the Systematic Coding of Text, Talk, and Other Verbal Data_](https://wac.colostate.edu/books/practice/codingstreams/) has proven to be an indispensable resource for researchers in the fields of rhetoric, composition, technical communication wishing to conduct qualitative and/or mixed method research into language use, including textual and verbal data. Geisler and Swarts (2019) lead practitioners through the entire process of qualitative or mixed-methods coding design, including the identification of research constructs, the creation of coding books, the application and validation of qualitative codes, data analysis, and reporting results.
    
Geisler and Swarts (2019) present a suite of statistical tests for validating codebooks (Cohen's Kappa) and determining the significance of numerical results (e.g., chi2 goodness of fit or chi2 homogeneity tests). The QC Utils app offers an online demonstration of key metrics discussed in Geisler and Swarts (2019) for teaching purposes and small-scale data analysis. As a pedagogical tool, QC Utils is best used in conjunction with introductory lessons on qualitative or mixed methods coding analytics, allowing students to view the results of statistical processing without intensive data preparation.
    
### Use Case
_Coding Streams_ already offers directive lessons and supplementary materials for codebook validation, significance testing, and multinomial logistic regression. These supplementary materials include videos and tutorials for conducting the aforementioned tests in Excel, AntConc (CITE), or MaxQDA. These lessons provide step-by-step instructions for utilizing statistical testing for research projects of any scale.
    
There is also an `R` app to compute multinomial logistic regression (CITE).

Outside of the ecosystem of resources provided by Geisler and Swarts (2019), there are other statistical processing libraries (like those underpinning `QC Utils`). For example, Ken Harmon (n.d.) offers a streamlit app that handles significance and regression testing: [https://github.com/harmkenn/python-stat-tools](https://github.com/harmkenn/python-stat-tools).

 `QC Utils` does not intend to replace materials associated with Geisler and Swarts (2019), nor is its goals the same as statistical processing apps such as Harmon (n.d.). `QC Utils` instead offers:
    
- an online demonstration of the keys tests in Geisler and Swarts (2019) situated within the disciplinary context of rhetoric, composition, writing studies, and technical communication
- a low friction computational tool that enables quick calculations through text input or file uploads
- a Python adaptation of Geisler and Swarts (2019) that connects its lessons to another, popular statistical programming language and provides example Python code so that people can take this work offline.
- additional statistical metrics not discussed in Geisler and Swarts (2019) but applicable
- trainable and downloadable machine learning classifiers (text and tabular data)
    
    """)
    
    st.markdown("""
    ### How to use `QC Utils`
    
The first step is to determine which test(s) is appropriate for your data (see Geisler and Swarts, 2019, pp. XX-XX).
    
`QC Utils` metrics can be divided into validation, significance testing, and regression tasks:
        
<dl>
<dt>Validation Tests</dt>
 <dd>Validation tests measure the agreement between two raters, thus indicating the robustness of the coding scheme (see also Boetgger and Palmer, 2019). </dd>
 <dd>If you need to calculate the inter-rater agreement between 2 coders, use the Validation Tests.</dd>
 <dd>Validation metrics include Cohen's Kappa and Krippendorff's Alpha.</dd>
<dt>Significance Tests </dt>
<dd>Significance tests reveal the relationships between variables.</dd>
<dd>Significance packaged in `QC Utils` include chi2 goodness of fit and chi2 test of homogeneity.</dd>
<dd>If you are seeking to understand how observed values compare with expected values, use the chi2 goodness of fit test. If you are seeking to understand if two distributions are significantly similar or different, use the chi2 test of homogeneity.</dd>
<dt>Regression Tests</dt>
<dd>Regression modeling endeavors to explain and/or predict the influence of independent variables on a dependent or predictor variable variable (McNulty, 2021; Massaron and Boschetti, 2016; Geisler and Swarts, 2019). </dd>
<dd>Regression tests packaged with `QC Utils` include Linear Regression and Logistic Regression for tabular and textual data. Both Linear and Logistic Regression utilties can be used to surface the effects of independent variables on a predictor variable and classify data.</dd>
<dd>If you wish to model the relationships of independent variables to continuous predictor variables, use Linear Regression. If you wish to understand the relationship of independent variables on categorical data, use Logistic Regression Tabular. If you wish to create a Logistic Regression text classifier to predict codes on new streams of language data, use Logistic Regression Classification Text.</dd>
</dl>

`QC Utils` can accept text and numerical inputs and file uploads (.csv, .xls, and .xlsx). See individual metric pages for input options.
    """,unsafe_allow_html=True)
    
    st.markdown("### References, Code Consulted, Software Libraries Used")
    st.write("""
          
Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.

Boettger, R. K., & Palmer, L. A. (2010). Quantitative content analysis: Its use in technical communication. IEEE transactions on professional communication, 53(4), 346-357.

Brownlee, J. (June 15 2018). A Gentle Introduction to the Chi-Squared Test for Machine Learning. Retrieved from https://machinelearningmastery.com/chi-squared-test-for-machine-learning/.

Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and psychological measurement, 20(1), 37-46.

Creswell, J. W. (2009). Research design: Qualitative, quantitative, and mixed methods approaches. Sage publications.

Cummings, J. R., Wen, H., Ko, M., & Druss, B. G. (2013). Geography and the Medicaid mental health care infrastructure: implications for health care reform. JAMA psychiatry, 70(10), 1084-1090.

Davidson, R., & MacKinnon, J. G. (2004). Econometric theory and methods (Vol. 5). New York: Oxford University Press.

Geisler, C., & Swarts, J. (2019). Coding streams of language: Techniques for the systematic coding of text, talk, and other verbal data. Ft. Collins, CO: WAC Clearinghouse.

Green, W. H. (2003). Econometric analysis. Pearson.

Greenfield, B., Melissa Henry, Margaret Weiss, Sze Man Tse, Jean-Marc Guile, Geoffrey Dougherty,  Xun Zhang, Eric Fombonne, Eric Lis,Sam Lapalme-Remis, Bonnie Harnden, (2008). Previously suicidal adolescents: predictors of six-month outcome. Journal of the Canadian Academy of Child and Adolescent Psychiatry, 17(4), 197.

Hart-Davidson, William. (2014). “Using Cohen’s Kappa to Gauge Interrater Reliability.” Education, 10:44:25 UTC. https://www.slideshare.net/billhd/kappa870.

Hemmert, G. A., Schons, L. M., Wieseke, J., & Schimmelpfennig, H. (2018). Log-likelihood-based pseudo-R 2 in logistic regression: deriving sample-sensitive benchmarks. Sociological Methods & Research, 47(3), 507-531.

Krippendorff, K. (2011). Computing Krippendorff's Alpha-Reliability. Retrieved from
https://repository.upenn.edu/asc_papers/43 

Lichman, M. (2013). UCI Machine Learning Repository [https://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

Lowry, R. (2014). Concepts and applications of inferential statistics.

Massaron, L., & Boschetti, A. (2016). Regression analysis with Python. Packt Publishing Ltd.

McFadden, Daniel . 1979. “Quantitative Methods for Analysing Travel Behaviour of Individuals: Some Recent Developments.” Pp. 279–318 in Behavioural Travel Modelling, edited by Hensher, D. A., Stopher, P. R.. London, UK: Croom Helm.

McHugh, Mary L. (2012 October 15). “Interrater Reliability: The Kappa Statistic.” Biochemia Medica 22, no. 3: 276–82.

McNulty, K. (2021). Handbook of regression modeling in people analytics: With examples in R and Python.

Montgomery, D. C., & Peck, E. A. (1992). Introduction to Linear Regression Analysis., 2nd edn.(John Wiley & Sons: New York.).

Multinomial Logistic Regression for Categorically-Coded Verbal Data. (n.d.). [Software]. Retrieved from https://shiny.chass.ncsu.edu/codingstreams/.

Norton, E. C., Dowd, B. E., & Maciejewski, M. L. (2019). Marginal effects—quantifying the effect of changes in risk factors in logistic regression models. Jama, 321(13), 1304-1305.

One-Off Coder. (2019). Data Science Topics. Retrieved from https://datascience.oneoffcoder.com.

Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions, Statistics and Probability Letters, 33 (1997) 291-297

Pearson, K. (1900). X. On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 50(302), 157-175.

Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E. (2011). Scikit-Learn: Machine Learning in Python. the Journal of machine Learning research, 12, 2825-2830.

scikit-learn developersA. (2021). sklearn.metrics.cohen_kappa_score. Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html.

scikit-learn developersB. (n.d.). Common pitfalls in the interpretation of coefficients of linear models. Retrieved from https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-download-auto-examples-inspection-plot-linear-model-coefficient-interpretation-py.

scipy Community. (2022). scipy.stats.chisquare. Retrieved from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html.

scipy Community. (2022). scipy.stats.chi2_contingency. Retrieved from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html.

Scott, William A. “Reliability of Content Analysis: The Case of Nominal Scale Coding.” The Public Opinion Quarterly 19, no. 3 (1955): 321–25.

Seabold, S., & Perktold, J. (2010). statsmodels: Econometric and statistical modeling with python. In 9th Python in Science Conference.

Sharpe, Donald (2015) "Chi-Square Test is Statistically Significant: Now What?," Practical Assessment,
Research, and Evaluation: Vol. 20 , Article 8.
DOI: https://doi.org/10.7275/tbfa-x148 

Streamlit. (May 2020). How to download a trained model. Retrieved from https://discuss.streamlit.io/t/how-to-download-a-trained-model/2976.

Szumilas, M. (2010). Explaining odds ratios. Journal of the Canadian academy of child and adolescent psychiatry, 19(3), 227.

Virtanen, P., Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt & SciPy 1.0 Contributors. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature methods, 17(3), 261-272
""")

main()

