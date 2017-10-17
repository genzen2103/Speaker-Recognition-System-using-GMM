# Speaker-Recognition-System-using-GMM
System for identifying speaker from given speech signal using MFCC,LPC features and Gaussian Mixture Models

Here we have two systems which are desinged for speaker recognition, One uses the popular MFCC features while the other does the same using Linear Predictive Coeficients.

For MFCC-GMM system, implementation is completely python based.
PyaudioAnalysis library has been used for feature extraction of voiced reagion of speaker
GMM modeling is done using sklearn package in python

For LPCC-GMM system implementation is partly in Scilab and python
For feature extraction Scilab script is used and GMM training is done using sklearn

Various experiments done for hyderparameter optimization can be found in report.
