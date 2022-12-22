An NLP model to predict the future success of early-stage companies using their own description as input feature. 
The data contains 5843 early-stage companies where 2156 of them are labelled as successful and the remaining 3687 are labelled as unsuccessful.
Companies that have more than USD $500M valuation either through an IPO (initial public offering), M&A (merger and acquisition) 
or large funding round (more than $150M funding)
are labelled as successful (1), whereas companies that raised more than $4M but less than $10M which
were founded between 2010 and 2016 are labelled as unsuccessful (0).
The descriptions have stopwords removed and padded into sequences. 
A neural network with three layers is then used with a training/validation split of 0.7.
The model returns an accuracy of 89%, a precision of 89% and a recall of 79%. 
