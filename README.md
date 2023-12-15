# Predicting Startup Success from Website Descriptions

## Overview

**Venture capital (VC) invests in early-stage companies believed to have long term growth potential.** This project explores the intriguing question: **"Can the self-description of early-stage companies on their websites serve as a predictive signal for venture capitalists to classify their future success?"** It aims to provide a data-driven approach to evaluate startup potential using machine learning techniques. Thanks to Vela Partners, we have a data set consisting of 5843 companies, 2156 are labeled as successful and 3687 are labeled as unsuccessful. Companies with more than USD $500M valuation through an IPO (initial public offering), M&A (merger and acquisition) or large funding round (more than $150M funding) are labeled as successful (1), whereas companies that raised more than $4M but less than $10M, founded between 2010 and 2016 are labelled as unsuccessful (0). 

In this notebook, we build a classification model using a deep learning approach by learning on the companies' self-description. **The model is simple, taking about 1 minute to train, yet it provides promising predictions.** Through cross-validation in 5 folds, the models consistently return an average **F1 score of 85%, recall of 81%, precision of 88% and accuracy of 89%.** For comparison, a random classifier as a baseline model returns a F1 score of 42%, recall of 48%, precision of 37% and accuracy of 48%. **These results suggest that the model is extremely effecitve in predicting and classifying the outcomes of startups founded between 2010 and 2016.** 

## Introduction

Investing in early-stage companies is often risky, requiring careful examinations of all available information by VCs. Platforms like TechCruch or Crunchbase can be utilized to scout potential companies aligning with specific VCs’ investment strategies and interests. New and young companies frequently turn to VCs for initial funding to scale their business in exchange for equity. If the business manage to grow, the rewards for the VCs could be substantial. However, it has been estimated 3 out of 4 venture-backed startups fail [(1)](https://www.wsj.com/articles/SB10000872396390443720204578004980476429190), making it crucial for VCs to avoid unsuccessful investment. While each VC has unique portfolio strategies, leveraging rapidly emerging machine learning methods for data analysis and portfolio mangement is increasingly common. [(2)](https://francesco-ai.medium.com/artificial-intelligence-and-venture-capital-af5ada4003b1).

Machine learning can support VCs in various ways, such as optimising portfolio management, identifying potential business partners and competitors [(3)](https://github.com/velapartners/weave), [(4)](https://github.com/velapartners/midas_touch_v2), creating startup valuation models [(5)](http://cs230.stanford.edu/projects_fall_2020/reports/55791766.pdf), [(6)](https://doi.org/10.1007/978-3-030-72113-8_12), developing recommendation systems to match startups with VCs [(7)](https://doi.org/https://doi.org/10.1016/j.ins.2019.11.045). **Machine learning methods provide data-driven non-obvious insights for investment decisions**, potentially highlighting continuation bias in decision-making. [(8)](https://techcrunch.com/2015/09/24/the-surprising-bias-of-venture-capital-decision-making/),[(9)](https://doi.org/10.1177/0971355713490818),[(10)](https://doi.org/http://dx.doi.org/10.2139/ssrn.3163955),[(11)](https://doi.org/http://dx.doi.org/10.2139/ssrn.4135861).

**In this project, we focus on developing a classifier which was deep learned on a subset of context from 6000 companies' website. The classifier predicts the future of a company into 1 (successful) and 0 (unsuccessful) given statements scraped from its website.** 

## Method

### Data Processing

The primary feature used for the classification algorithm is the companies' descriptions. These are processed to remove stopwords like 'am', 'at', 'about', which carry minimal meaning. The first word of company names is also considered as a stopword, as it often does not capture the essence of the company's nature.

The successful and unsuccessful data is then shuffled together, with 70% of them split into training and the remaining 30% used for validation. The training sentences' vocabulary is tokenized using the TensorFlow Keras package, with an 'OOV' token to replace out-of-vocabulary words. 

The dimension of the dense embedding layer of the model is chosen to be 8. All description of the startups are padded into 85 words with post padding strategy.

### Deep Learning Model

The neural consists of an embedding layer, followed by ’GlobalAveragePooling1D’ layer, a dense layer with 163 neurons using softmax activation, a dense layer with 57 neurons using sigmoid activation, an ordinary dense layer with 17 neurons, and at last a dense layer with 2 neurons using softmax activation. The model is compiled with the ”sparse categorical crossentropy” loss function, and an ’Adam’ optimizer.

## Result
In this notebook, we build a classification model using a deep learning approach by learning on the companies' self-description. The model is inherently simple, it takes about 1 minute to train the model. Yet, it is able to give promising predictions. Through cross-validating into 5 folds, the models consistently return an average F1 score of 85%, recall of 81%, precision of 88% and accuracy of 89%. When compared with a baseline random classifier which would return 42% F1 score, 37% precision, 48% recall and 48% accuracy, this deep learnred models perform significantly better than the baseline model.

## Discussion

There are various approaches for these prediction method [(15)](https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/HENGSTBERGER_THOMAS_01822754.pdf),[(16)](https://doi.org/10.1109/ACCESS.2019.2938659),[(17)](https://doi.org/https://doi.org/10.1016/j.jfds.2021.04.001),[(18)](https://doi.org/https://doi.org/10.1016/j.jfds.2021.04.001),[(19)](https://doi.org/10.48550/ARXIV.2210.14195),[(20)](https://github.com/velapartners/moneyball-temporal-v1). including quantizing founders’ background[(21)](https://github.com/velapartners/moneyball-v2/blob/main/Vela_Partners_Project.pdf) and employing natural language processing methods via BERT transfer learning or Word2Vec [(23)](https://github.com/velapartners/maverick),[(24)](https://github.com/velapartners/twins-v2),[(25)](https://github.com/velapartners/moneyball-v3).

## Dependencies
Libraries: pandas, numpy, matplotlib, seaborn, tensorflow, sklearn, scikeras
