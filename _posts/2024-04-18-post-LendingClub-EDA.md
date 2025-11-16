---
title: "Decoding Borrower Behavior"
date: 2024-04-18T15:34:30-04:00
categories:
  - Post
tags:
  - EDA
---

Exploratory data analysis or EDA is the foundation of every Data Science and Machine Learning project because it helps you deeply understand the data before building any model. Through EDA, you uncover patterns, detect anomalies, identify missing or noisy data, and validate assumptions. This early insight shapes the entire workflow—from choosing the right features and algorithms to defining preprocessing steps—ultimately ensuring the model is built on clean, meaningful, and well-understood data. Without EDA, even the most advanced ML techniques can fail due to hidden issues in the dataset.

Here is a case study for **Credit Risk Analysis** where I explore what leads to Loan Defaults.

**The Problem statement** 
When a company receives a loan application,the company has to make a decision for loan approval based on the applicant’s profile. 
Two types of risks are associated with the bank’s decision:
1. If the applicant is likely to repay the loan, then not approving the loan results in a loss of business to the company. 
2. If the applicant is not likely to repay the loan, i.e. he/she is likely to default, then approving the loan may lead to a financial loss for the company. 

**Approach**
In this case study we will try to understand how consumer attributes and loan attributes influence the tendency of default.If we are able to identify these risky loan applicants, then such loans can be reduced thereby cutting down the amount of credit loss.In other words, a company wants to understand the driving factors (or driver variables) behind loan default, i.e. the variables which are strong indicators of default.

**Dataset**
The data used for this exercise is stored in "Loan.csv" and it is available in the github repo linked below .

**Tools and Libraries used**
- For data and numeric analysis
  1. Pandas
  2. Numpy

- For visualization
1. Matplotlib
2. Seaborn

**Method**
- STEP 1: Understanding the data

We begin the process of EDA by answering some of the questions like:
1. What is the shape of the data (number of rows & columns)
2. Are there any summary/header/footer rows?
3. Presence of null values in rows/columns?
4. What kind of data is available,from a business perspective?

We found was there are mainly 3 types of data present in the dataset.
1. Consumer demographics: such as annual_inc, home_ownership, emp_length
2. Loan attributes: such as int_rate, loan_amt,term etc
3. Data/columns which are calculated only after loan is approved:total_payment, revol_bal etc

- STEP 2: Data cleaning and Manipulation

In this step We address various data quality issues by:
1. Indentifying and imputing missing values
2. Removing data redundancies
3. Filtering/assumptions on data based on business knowledge
4. Standardising Values (fixing datatypes,date & string manipulation,binning etc)
5. Outlier treatment
6. Creating some derived metrics

- STEP 3: Exploratory Data Analysis

Next we understand our target variable "Loan status" and begin EDA in the following order:
1. Univariate - numerical variables
2. Univariate - categorical variables
3. Segmented univariate analysis
4. Bivariate analysis
5. Multivariate analysis

Based on the analysis we were able to indentify some core risk indicators which determined if a Loan gets approved or not.

**RESULTS**
The core risk indicators from this study were -

Risky Consumer attributes :
1. Annual income: Annual income in the ranges of 35k to 60k saw higher defaults.
2. Loan to income ratio: Borrowers with high loan to income ratio tend to default more as payback power is less.
3. Home ownership: Borrowers who had rented accommodation or a mortgage collectively had higher default rates.
4. State: Borrowers from large urban states were prone to defaulting.

Risky Loan attributes:
1. Loan amount: Higher the loan amount, higher the chances of default.
2. Rate of interest: Interest rate of 13.5% and above lead to higher defaults.
3. Term: A 60-month loan tenure was riskier than 36 months
4. EMI: Higher EMIs lead to higher chances of defaulting
5. Verification Status: Unexpectedly, borrowers with income source as "verified" defaulted more compared to the ones  without verification. 
  *This indicated that the verification procedure followed by the company was faulty*

To understand and visualize how these insights were drawn, please visit :
Github [link](https://github.com/nehaseng/LendingClubCaseStudy)
