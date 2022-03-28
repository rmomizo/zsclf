# QC Utils
[https://share.streamlit.io/rmomizo/zsclf/main/kappa_streamlit.py](https://share.streamlit.io/rmomizo/zsclf/main/kappa_streamlit.py)

## Overview
QC Utils demos key metrics from Cheryl Geisler and Jason Swarts' (2019) Coding Streams of Language: Techniques for the Systematic Coding of Text, Talk, and Other Verbal Data.

QC Utils serves as wrapper for the Python statistical processing libraires scikit-learn (Pedrogosa et al., 2013), The Natural Language Toolkit (NLTK),statsmodels, and scipy (Virtanen et al., 2020) built and hosted by Streamlit.

With QC Utils, you can run:

Cohen's Kappa inter-rater agreement tests for two coders
Krippendoff's Alpha inter-rater agreement tests for two coders
chi2 goodness of fit test for single samples
chi2 test of homogeneity for two samples
linear regression (tabular data)
logistic regression classification (text and tabular data)
Introduction
Cheryl Geisler and Jason Swarts' (2019) guide to mixed-methods coding research, Coding Streams of Langauge: Techniques for the Systematic Coding of Text, Talk, and Other Verbal Data has proven to be an indispensable resource for researchers in the fields of rhetoric, composition, technical communication wishing to conduct qualitative and/or mixed method research into language use, including textual and verbal data. Geisler and Swarts (2019) lead practitioners through the entire process of qualitative or mixed-methods coding design, including the identification of research constructs, the creation of coding books, the application and validation of qualitative codes, data analysis, and reporting results.

Geisler and Swarts (2019) present a suite of statistical tests for validating codebooks (Cohen's Kappa) and determining the significance of numerical results (e.g., chi2 goodness of fit or chi2 homogeneity tests). The QC Utils app offers an online demonstration of key metrics discussed in Geisler and Swarts (2019) for teaching purposes and small-scale data analysis. As a pedagogical tool, QC Utils is best used in conjunction with introductory lessons on qualitative or mixed methods coding analytics, allowing students to view the results of statistical processing without intensive data preparation.

## Use Case
Coding Streams already offers directive lessons and supplementary materials for codebook validation, significance testing, and multinomial logistic regression. These supplementary materials include videos and tutorials for conducting the aforementioned tests in Excel, AntConc (CITE), or MaxQDA. These lessons provide step-by-step instructions for utilizing statistical testing for research projects of any scale.

There is also an R app to compute multinomial logistic regression(https://shiny.chass.ncsu.edu/codingstreams/) created by Emily Griffith of NC State University.

Outside of the ecosystem of resources provided by Geisler and Swarts (2019), there are other statistical processing libraries (like those underpinning QC Utils). For example, Ken Harmon (n.d.) offers a streamlit app that handles significance and regression testing: https://github.com/harmkenn/python-stat-tools.

QC Utils does not intend to replace materials associated with Geisler and Swarts (2019), nor is its goals the same as statistical processing apps such as Harmon (n.d.). QC Utils instead offers:

- an online demonstration of the keys tests in Geisler and Swarts (2019) situated within the disciplinary context of rhetoric, composition, writing studies, and technical communication
- a low friction computational tool that enables quick calculations through text input or file uploads
- a Python adaptation of Geisler and Swarts (2019) that connects its lessons to another, popular statistical programming language and provides example Python code so that people can take this work offline.
- additional statistical metrics not discussed in Geisler and Swarts (2019) but applicable
- trainable and downloadable machine learning classifiers (text and tabular data)
## How to use QC Utils
The first step is to determine which test(s) is appropriate for your data (see Geisler and Swarts, 2019, pp. XX-XX).

QC Utils metrics can be divided into validation, significance testing, and regression tasks:

### Validation Tests
Validation tests measure the agreement between two raters, thus indicating the robustness of the coding scheme (see also Boetgger and Palmer, 2019).
If you need to calculate the inter-rater agreement between 2 coders, use the Validation Tests.
Validation metrics include Cohen's Kappa and Krippendorff's Alpha.
### Significance Tests
Significance tests reveal the relationships between variables.
Significance packaged in `QC Utils` include chi2 goodness of fit and chi2 test of homogeneity.
If you are seeking to understand how observed values compare with expected values, use the chi2 goodness of fit test. If you are seeking to understand if two distributions are significantly similar or different, use the chi2 test of homogeneity.
### Regression Tests
Regression modeling endeavors to explain and/or predict the influence of independent variables on a dependent or predictor variable variable (McNulty, 2021; Massaron and Boschetti, 2016; Geisler and Swarts, 2019).
Regression tests packaged with `QC Utils` include Linear Regression and Logistic Regression for tabular and textual data. Both Linear and Logistic Regression utilties can be used to surface the effects of independent variables on a predictor variable and classify data.
If you wish to model the relationships of independent variables to continuous predictor variables, use Linear Regression. If you wish to understand the relationship of independent variables on categorical data, use Logistic Regression Tabular. If you wish to create a Logistic Regression text classifier to predict codes on new streams of language data, use Logistic Regression Classification Text.
QC Utils can accept text and numerical inputs and file uploads (.csv, .xls, and .xlsx). See individual metric pages for input options.

## References, Code Consulted, Software Libraries Used
Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.

Boettger, R. K., & Palmer, L. A. (2010). Quantitative content analysis: Its use in technical communication. IEEE transactions on professional communication, 53(4), 346-357.

Brownlee, J. (June 15 2018). A Gentle Introduction to the Chi-Squared Test for Machine Learning. Retrieved from https://machinelearningmastery.com/chi-squared-test-for-machine-learning/.

Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and psychological measurement, 20(1), 37-46.

Creswell, J. W. (2009). Research design: Qualitative, quantitative, and mixed methods approaches. Sage publications.

Cummings, J. R., Wen, H., Ko, M., & Druss, B. G. (2013). Geography and the Medicaid mental health care infrastructure: implications for health care reform. JAMA psychiatry, 70(10), 1084-1090.

Davidson, R., & MacKinnon, J. G. (2004). Econometric theory and methods (Vol. 5). New York: Oxford University Press.

Geisler, C., & Swarts, J. (2019). Coding streams of language: Techniques for the systematic coding of text, talk, and other verbal data. Ft. Collins, CO: WAC Clearinghouse.

Green, W. H. (2003). Econometric analysis. Pearson.

Greenfield, B., Melissa Henry, Margaret Weiss, Sze Man Tse, Jean-Marc Guile, Geoffrey Dougherty, Xun Zhang, Eric Fombonne, Eric Lis,Sam Lapalme-Remis, Bonnie Harnden, (2008). Previously suicidal adolescents: predictors of six-month outcome. Journal of the Canadian Academy of Child and Adolescent Psychiatry, 17(4), 197.

Hart-Davidson, William. (2014). “Using Cohen’s Kappa to Gauge Interrater Reliability.” Education, 10:44:25 UTC. https://www.slideshare.net/billhd/kappa870.

Hemmert, G. A., Schons, L. M., Wieseke, J., & Schimmelpfennig, H. (2018). Log-likelihood-based pseudo-R 2 in logistic regression: deriving sample-sensitive benchmarks. Sociological Methods & Research, 47(3), 507-531.

Krippendorff, K. (2011). Computing Krippendorff's Alpha-Reliability. Retrieved from https://repository.upenn.edu/asc_papers/43

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

Sharpe, Donald (2015) "Chi-Square Test is Statistically Significant: Now What?," Practical Assessment, Research, and Evaluation: Vol. 20 , Article 8. DOI: https://doi.org/10.7275/tbfa-x148

Streamlit. (May 2020). How to download a trained model. Retrieved from https://discuss.streamlit.io/t/how-to-download-a-trained-model/2976.

Szumilas, M. (2010). Explaining odds ratios. Journal of the Canadian academy of child and adolescent psychiatry, 19(3), 227.

Virtanen, P., Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt & SciPy 1.0 Contributors. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature methods, 17(3), 261-272
