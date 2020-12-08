# Recurrent Neural Networks (RNN) for Natural Language Processing (NLP)
### IMDB Review Sentiment Classification

This project uses movie reviews from the IMDB dataset to build a sentiment classification model.  The reviews are labeled as either positive (1) or negative (0). The data can be loaded directly from the tensorflow package using the code below.

```
import tensorflow as tf
import tensorflow_datasets as tfds
dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
```

The data can also be found and downloaded here, if preferred:
- http://ai.stanford.edu/~amaas/data/sentiment/

Our goal is to correctly predict review sentiment based on the text in each review.  Specifically, we aim to build a model that maximizes overall accuracy.

**Problem Description:**

The dataset that we use to build and test our model contains 50,000 text reviews, with 25,000 each in the train and test datasets.  Our target is binary (positive, negative) and class-balanced.  To increase computational speed, we randomly sample 5,000 reviews for training and 1,000 for validation and testing (500 each).

To serve as a benchmark, we start by creating a rule-based sentiment model, based on the number of positive and negative words in each review.  Then, to measure improvement, we make a RNN model with LSTM (long short-term memory) to find and learn long-range word dependencies across a sentence (and review).  In the end, we compare the accuracy of our simple rule-based model with our more complicated RNN model.
