---
layout : post
title: "📨 Building a Spam–Ham Classifier Using NLP"
description: "A beginner-friendly walkthrough of how I built a machine learning model that classifies SMS messages as spam or ham using Python, NLP, and TF-IDF."
date: 2024-04-28T15:34:30-04:00
tags:tags: [machine-learning, nlp, python, spam-classifier, beginner-friendly]
---

# 📨 Building a Spam–Ham Classifier Using NLP  
*A simple, story-driven explanation of how I taught a machine to detect spam.*

If you're stepping into the world of Natural Language Processing (NLP), there's one project that almost everyone builds at least once—the **Spam vs Ham Classifier**.

Think of it as the “Hello World!” of NLP: simple, practical, and the perfect confidence-boosting first project.

In this post, we’ll walk through how I built a simple yet complete spam detection system from scratch. No prior ML knowledge needed—just curiosity and an interest in how machines understand text.

We’ll go step by step:

- Reading the data  
- Preprocessing the text  
- Converting text → numerical features  
- Training an ML model  
- Evaluating the model  
- Predicting whether a new message is SPAM or HAM 

Let’s begin.

---

## 📘 Table of Contents
1. [Dataset Overview](#dataset-overview)  
2. [Preprocessing Pipeline](#preprocessing-pipeline)  
3. [Converting Text into Features](#converting-text-into-features)  
4. [Model Building](#model-building)  
5. [Model Evaluation](#model-evaluation)  
6. [Prediction on New Messages](#prediction-on-new-messages)  
7. [Final Thoughts](#final-thoughts)

---

## ✉️ Dataset Overview {#dataset-overview}

We use a simple SMS dataset containing two columns:

- **label** → spam / ham  
- **message** → the actual SMS text 

The goal? Train a model that learns the patterns inside these messages. 

```python
import pandas as pd

df = pd.read_csv("spam.csv", encoding="ISO-8859-1")
df = df[['v1','v2']]
df.columns = ['label','message']
df.head()
```

At this point, it's just raw text. To a machine, this means nothing yet.

## 🧹 Preprocessing Pipeline
{: #preprocessing-pipeline }

<div class="tip-box">
This is the step where we “teach” the machine how to read text.  
We remove noise, standardize patterns, and keep only the meaningful parts.
</div>

We perform:

| Step | Purpose |
|------|---------|
| **Lowercasing** | Standardizes words (WINNER → winner) |
| **Remove punctuation** | Eliminates noise like ! ? , . |
| **Tokenization** | Splits sentences into words |
| **Stopword removal** | Removes frequent but useless words (“the”, “is”, “at”) |
| **Stemming** | Reduces words to root forms (winning → win) |

### 🧽 Cleaning Function

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(msg):
    msg = msg.lower()
    tokens = nltk.word_tokenize(msg)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)

df['clean_msg'] = df['message'].apply(clean_text)
df.head()
```

Original:  "WINNER! Claim your free prize now!!!"
Cleaned:   "winner claim free prize"


<div class="example-box"> <strong>Before:</strong> WINNER! Claim your free prize now!!!

<strong>After:</strong>
winner claim free prize
</div>

After this stage, “WINNER! Claim your free prize now!” becomes something like:

>“winner claim free prize”

Now our model can focus on the essence, not the noise.

## 🔍 Some Data Analysis

Before modeling, it's always good to peek into the data:

- How many messages are spam vs ham?

- What words appear frequently?

-Are spam messages typically longer?

These simple checks help validate dataset quality and give intuition about what the model will learn.

```python
df.label.value_counts()
```

Often, the dataset is imbalanced—**ham messages dominate**—so evaluation metrics matter.

## Converting Text into Features {#converting-text-into-features}
Machines love numbers, not words.
So we convert text into numerical vectors using TF-IDF, which measures:
- how important a word is
- relative to all other words
- within all messages

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_msg']).toarray()

y = df['label'].map({'ham':0, 'spam':1})

```
Now every SMS message becomes a numerical vector or simply put a row of numbers representing word importance.
This is the step where text finally becomes machine-learnable.

## Model Building{: #model-building }
<div class="tip-box"> For text classification, Naive Bayes is often incredibly strong despite being simple. </div>

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

```
It learns classic spam patterns like:
* “win”, “free”, “urgent”
* lots of uppercase
* promotional tone

## 📈 Model Evaluation {#model-evaluation}

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

accuracy_score(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

```
You’ll usually get **95%+ accuracy** —text classification works surprisingly well even with simple models!

A confusion matrix helps check if the model mistakes spam for ham (dangerous) or the other way around (annoying but safe).

<div class="diagram-box">
graph TD
    A[Predicted Spam] -->|Correct| B[Actual Spam]
    A -->|Incorrect| C[Actual Ham]

    D[Predicted Ham] -->|Correct| C
    D -->|Incorrect| B
</div>

## 🧪 Prediction on New Messages {#prediction-on-new-messages}

``` python
msg = ["Congratulations! You have won a $1000 gift card. Click to claim."]
clean_msg = clean_text(msg[0])
vector = tfidf.transform([clean_msg])
model.predict(vector)
```

``` makefile
Output: [1] → SPAM
```

``` python
model.predict(tfidf.transform([clean_text("Are we meeting today?")]))
```
```makefile 
Output: [0] → HAM
```
## Final Thoughts {#final-thoughts}
This project might be considered “classical ML,” but it teaches the core concepts behind almost every NLP system today:

- cleaning text
- representing text numerically
- training a model
- evaluating performance
- making real predictions

These foundations never change, even in the age of transformers and LLMs.
If you're just starting your ML journey, this project is a perfect confidence booster.
And if you’re already experienced, it's a great warm-up before diving into deep learning-based NLP (like transformers).