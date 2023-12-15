# Predicting Startup Success from Website Descriptions

## Overview

Venture capital (VC) invests in early-stage companies that are believed to have long term growth potential. This project explores the intriguing question: "Can the self-description of early-stage companies on their websites serve as a predictive signal for venture capitalists to classify their future success?" It aims to provide a data-driven approach to evaluate startup potential using machine learning techniques. Thanks to Vela Partners, we are provided with a data set consisting of 5843 companies, 2156 are labelled as successful and 3687 are labelled as unsuccessful. Companies that have more than USD $500M valuation either through an IPO (initial public offering), M&A (merger and acquisition) or large funding round (more than $150M funding) are labelled as successful (1), whereas companies that raised more than $4M but less than $10M which were founded between 2010 and 2016 are labelled as unsuccessful (0). 

In this notebook, we build a classification model using a deep learning approach by learning on the companies' self-description. The model is inherently simple, it takes about 1 minute to train the model. Yet, it is able to give promising predictions. Through cross-validating into 5 folds, the models consistently return an average F1 score of 85%, recall of 81%, precision of 88% and accuracy of 89%. For comparison, a random classifier as a baseline model returns a F1 score of 42%, recall of 48%, precision of 37% and accuracy of 48%. The results suggest that this model is extremely effecitve in predicting and classifying the outcomes of startups founded between 2010 and 2016. 

## Introduction

Investing in early-stage companies can often be risky on unproven companies and thus requires careful examinations on every information that is available to the VCs. One could utilize platforms like techcruch or crunchbase to scout potential companies that may align with specific VCs’ investment strategies and interests. New and young companies often turn to VCs for initial funding to scale their business in exchange for the companies’ own equities. If the business manage to grow, the rewards for the VCs could be huge. However, it has been estimated 3 out of 4 venture-backed startups fail [(1)](https://www.wsj.com/articles/SB10000872396390443720204578004980476429190), VCs have to bear in mind how to avoid unsuccessful investment due to the high failure rate of startups. While each VC has their unique portfolio strategies, it is only reasonable one would leverage the rapidly emerging machine learning methods to assist in data analysis and manage portfolio [(2)](https://francesco-ai.medium.com/artificial-intelligence-and-venture-capital-af5ada4003b1).

In general, machine learning can support VCs in various ways, including but not limited to i) optimising portfolio management, ii) identifying potential business partners and competitors [(3)](https://github.com/velapartners/weave), [(4)](https://github.com/velapartners/midas_touch_v2), iii) creating startup valuation models [(5)](http://cs230.stanford.edu/projects_fall_2020/reports/55791766.pdf), [(6)](https://doi.org/10.1007/978-3-030-72113-8_12), iv) creating recommendation systems to match startups with VCs [(7)](https://doi.org/https://doi.org/10.1016/j.ins.2019.11.045). Machine learning methods provide data-driven non-obvious insights for investment decisions, which may remind investors of potential continuation bias towards decisions made [(8)](https://techcrunch.com/2015/09/24/the-surprising-bias-of-venture-capital-decision-making/),[(9)](https://doi.org/10.1177/0971355713490818),[(10)](https://doi.org/http://dx.doi.org/10.2139/ssrn.3163955),[(11)](https://doi.org/http://dx.doi.org/10.2139/ssrn.4135861). Machine learning has also been applied on predicting the behaviours of various VCs, where attempts have been made to classify whether a startup would be invested by a certain group of VCs [(12)](),[(13)](),[(14)]().

In this paper, the author focus on leveraging machine learning classification algorithms to predict the future of a startup into 1 (successful) and 0 (unsuccessful). There are a variety of approaches on this prediction method [15]–[20]. For example, by quantizing founders’ background for classification
purposes, one could achieve a precision of about 0.8 [21], [22]. It has also been found that a few characteristics of founders such as team size, academic abilities and previous positions at different companies do have some predictive power for future outcomes [23]. Employing natural language processing methods via BERT transfer learning or Word2Vec could also be seen to study startups [24]–[26].


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
