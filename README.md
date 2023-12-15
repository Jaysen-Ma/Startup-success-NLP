# Predicting Startup Success from Website Descriptions

## Overview

This project explores the intriguing question: "Can the self-description of early-stage companies on their websites serve as a predictive signal for venture capitalists to classify their future success?" It aims to provide a data-driven approach to evaluate startup potential using machine learning techniques. 

## Contents

    ### Data Processing: Steps involved in preparing the data for analysis.
    ### Removing Stopwords: Techniques applied to clean the text data.
    ### Global Variables: Definition of global variables used in the notebook.
    ### Training-Validation Split: Methodology for splitting the dataset into training and validation sets.
    ### Tokenization, Sequences, and Padding: Processes involved in converting text data into a format suitable for machine learning models.

In this notebook, we build a classification model using a deep learning approach by learning on the companies' self-description. The model is inherently simple, it takes about 1 minute to train the model. Yet, it is able to give promising predictions. Through cross-validating into 5 folds, the models consistently return an average F1 score of 85%, recall of 81%, precision of 88% and accuracy of 89%. When compared with a baseline random classifier which would return 42% F1 score, 37% precision, 48% recall and 48% accuracy, this deep learnred models perform significantly better than the baseline model.

## Dependencies
    Libraries: [pandas, numpy, matplotlib, seaborn, tensorflow, sklearn, scikeras.]

## Contact
  For any queries or further discussions, feel free to contact me at chunhinma00@outlook.com
