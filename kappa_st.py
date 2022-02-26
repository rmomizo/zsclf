import streamlit as st 
import pandas as pd 
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from scipy.stats import chi2, chi2_contingency, chisquare

def chi_goodness():
    """
    
    """
    st.title('chi2 Goodness of Fit Test')
    st.write('This chi2 calculator assumes that your data consists of a single frequency distribution:')
    
    
    st.markdown("""
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
    1. Input the significant value (default/max value is .05)
    2. Copy the values of your sample and paste into the Sample text entry field and hit "Enter." 
    
    ❗By default, expected frequencies are equally likely. 
       """)
    significance = float(st.text_input('Input significance value (default is .05; max value is .1)', value='.05'))
    if int(significance) not <= .1:
        st.text('The maximum significance value is .1')
        pass
    col1 = st.text_input('Sample 1',value='37 75 98')
   
    s1 = [int(c) for c in col1.split()]
   
    chi, p_val = chisquare(s1)
    p = 1 - significance
    crit_val = chi2.ppf(p, len(s1)-1)
    st.subheader('Results')
    c1 = st.container()
    c2, c3, c4, c5 = st.columns(4)
    
    c1.metric('p-value', str(p_val))
    c2.metric('Dataset Length',str(len(s1)))
    c3.metric('degree of freedom',"{:.2f}".format(len(s1)-1)) 
    c4.metric('\n chi2 test statistic',"{:.5f}".format(chi)) 
    c5.metric('critical value',"{:.5f}".format(crit_val))
    st.write("For an extended discussion of using chi2 goodness of fit tests for qualitative coding, see [Geisler and Swarts (2019)](https://wac.colostate.edu/docs/books/codingstreams/chapter9.pdf)")
       

def chi():
    """
    Python code adapted from Brownlee (June 15, 2018)
    """
    st.title('chi2 Test of Homogeneity')
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
    1. Input the significant value (default/max value is .05)
    2. Copy the values of your first sample and paste into the Sample 1 text entry field and hit "Enter." 
    3. Copy the values for your second sample and paste into the Sample 2 text entry field and hit "Enter."
    ❗Samples 1 and Sample 2 must be numerical values. 
       """)
    significance = float(st.text_input('Input significance value (default/max value is .05)', value='.05'))
    col1 = st.text_input('Sample 1',value='10 20 30')
    col2 = st.text_input('Sample 2', value='10 15 25')

    s1 = [int(c) for c in col1.split()]
    s2 = [int(c) for c in col2.split()]
       
    chi, p_val, dof, ex = chi2_contingency([s1,s2], correction=False)
    p = 1 - significance
    crit_val = chi2.ppf(p, dof)
    st.subheader('Results')
    c1 = st.container()
    c2, c3, c4, c5 = st.columns(4)
    
    c1.metric('p-value', str(p_val))
    c2.metric('Dataset Length',str(len(s1)))
    c3.metric('degree of freedom',"{:e}".format(dof)) 
    c4.metric('\n chi2 test statistic',"{:.5f}".format(chi)) 
    c5.metric('critical value',"{:.5f}".format(crit_val))
    st.write("For an extended discussion of using chi2 tests for homogeneity for qualitative coding, see [Geisler and Swarts (2019)](https://wac.colostate.edu/docs/books/codingstreams/chapter9.pdf)")

def chi_file_upload():
    """
    Python code adapted from Brownlee (June 15, 2018)
    """
    st.title('chi2 Test of Homogeneity')
    st.write('This chi2 calculator assumes that your data is in the form of a contingency table:')
    
    
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
    2. Upload your contingency table as an .csv or .xlsx file. Make sure that the column names for your two samples are "sample 1" and "sample 2."
       """)
    significance = float(st.text_input('Input significance value (default/max value is .05)', value='.05'))
    
    uploaded = st.file_uploader('Upload your .csv or .xlsx file.')
    if uploaded != None:
        if uploaded.name.endswith('csv'):
            df = pd.read_csv(uploaded)
            s1 = [int(c) for c in df['sample 1']]
            s2 = [int(c) for c in df['sample 2']]
       
            chi, p_val, dof, ex = chi2_contingency([s1,s2], correction=False)
            p = 1 - significance
            crit_val = chi2.ppf(p, dof)
            
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
            crit_val = chi2.ppf(p, dof)

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

def kappa():
    st.title("Cohen's Kappa Calculator")   
    st.write("""
    1. Copy the codes of Coder 1 into the Coder 1 text entry field and hit "Enter." 
    2. Copy the codes for Coder 2 into the Coder 2 text entry field and hit "Enter."
    ❗ Make sure that the coding decisions between Coder 1 and Coder 2 are the same length.
       """)
    
    col1 = st.text_input('Coder 1',value='a a b')
    col2 = st.text_input('Coder 2', value='a a b')
 
    try:
        st.subheader('Results')
        c1, c2, c3 = st.columns(3)
        c1.metric('Dataset Length',str(len(col1.split())))
        c2.metric('Accuracy',str(accuracy_score(col1.split(),col2.split())))
        c3.metric('Kappa Score',str(cohen_kappa_score(col1.split(),col2.split())))

        labels = sorted(list(set(col1.split()+ col2.split())))
        indices = [str(label)+'_' for label in labels]
        st.write("Confusion Matrix:")
        st.table(pd.DataFrame(confusion_matrix(col1.split(),col2.split()),index=indices,columns=labels))
        st.caption('Note: Coder 1 is used as the baseline for evaluation.')
        st.markdown("For more an extended presentation on Cohen's Kappa see Hart-Davidson (2014), [Using Cohen's Kappa to Gauge Interrater Reliability](https://www.slideshare.net/billhd/kappa870)")
    except ValueError:
        st.markdown('<mark>Error: Data must be the same length</mark>', unsafe_allow_html=True)
    
def kappa_file_upload():
    st.title("Cohen's Kappa Calculator")   
    st.markdown("""
    Upload your .csv or .xlsx file. 
    
    Your files should feature the following format:
       """)
       
    dff = pd.DataFrame({'Coder 1':['a','a','b'],'Coder 2': ['a','a','b']})
    st.dataframe(dff)
    
    uploaded_file = st.file_uploader("Upload your data as .csv or .xlsx")
    
   
    if uploaded_file != None:
        if str(uploaded_file.name).endswith('csv'):
            df = pd.read_csv(uploaded_file)

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
            st.table(pd.DataFrame(confusion_matrix(col1,col2),index=indices,columns=labels))
            st.caption('Note: Coder 1 is used as the baseline for evaluation.')

            st.markdown("For more an extended presentation on Cohen's Kappa see Hart-Davidson (2014), [Using Cohen's Kappa to Gauge Interrater Reliability(https://www.slideshare.net/billhd/kappa870)")
            #except ValueError:
             #   st.markdown('<mark>Error: Data must be the same length</mark>', unsafe_allow_html=True)
        elif str(uploaded_file.name).endswith('xlsx'):
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
           st.table(pd.DataFrame(confusion_matrix(col1,col2),index=indices,columns=labels))
           st.caption('Note: Coder 1 is used as the baseline for evaluation.')

           st.markdown("For more an extended presentation on Cohen's Kappa see Hart-Davidson (2014), [Using Cohen's Kappa to Gauge Interrater Reliability](https://www.slideshare.net/billhd/kappa870)")
           #except ValueError:
            #   st.markdown('<mark>Error: Data must be the same length</mark>', unsafe_allow_html=True)
    
    
def main():
    
    st.sidebar.title("Cohen's Kappa and chi2 Calculator")
    st.sidebar.subheader("Calculate the inter-rater agreement between two coders using sklearn's `cohen_kappa_score` module or calculate the chi2 homogeneity of two samples with `scipy`")

    options = st.sidebar.selectbox('What metric would you like to apply?',("Cohen's Kappa","chi2 Goodness of Fit", "chi2 Homogeneity"))
    data_options = st.sidebar.selectbox('How would you like to submit your data?',("Copy and Paste","Upload .csv or .xlsx"))
    
 
    if options == "Cohen's Kappa" and data_options == "Copy and Paste":
        
        kappa()
    elif options == "Cohen's Kappa" and data_options == "Upload .csv or .xlsx":
       kappa_file_upload()
    elif options == 'chi2 Homogeneity' and data_options == "Copy and Paste":
        chi()
    elif options == 'chi2 Homogeneity' and data_options == "Upload .csv or .xlsx":
        chi_file_upload()
    elif options == 'chi2 Goodness of Fit' and data_options == "Copy and Paste":
       chi_goodness()
    else:
       chi_goodness_file_upload()

main()

with st.sidebar.expander("See References and Code Consulted"):
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

st.sidebar.write('© Ryan Omizo 2022')
