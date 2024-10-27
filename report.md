Chan Lee

MSDS 422

Final Project: OpenML German Credit data set

https://www.openml.org/search?type=data&status=active&id=31



## Executive Summary

The OpenML German Credit data set contains information for 1,000 individuals with each entry also having a classification of being either a good or bad credit risk. This project aims to develop a predictive model capable of classifying credit risk as well as to determine which factors influence creditworthiness.

## Research Objectives

Accurately assessing credit risk is essential for managing porfolios and minimizing defaults. Therefore, developing a predictive model to make data-driven decisions can allow the business to scale safely. This project has two primary research objectives:

* Gain a thorough understanding of key factors to credit risk
* Develop a classification model that can effectively predict credit risk

## Exploratory Data Analysis (EDA)

*Source code can be found in midpoint.ipynb*

The data set provided by OpenML is a collection of 20 attributes associated with 1000 individuals and their final risk assessment of either `good` or `bad`. A cursory look into the data shows that there are no missing values present for any of the entries.

<img src="/Users/chanlee/Library/Application Support/typora-user-images/Screenshot 2024-10-26 at 7.47.26 PM.png" alt="Screenshot 2024-10-26 at 7.47.26 PM" style="zoom:50%;" />

#### Some feature engineering

Running a correlation analysis on the numeric columns shows that `credit_amount` (the dollar amount of the loan) and `duration` (the duration of the loan in months) are highly correlated with a coefficient of ~0.8.

![Screenshot 2024-10-26 at 7.50.18 PM](/Users/chanlee/Library/Application Support/typora-user-images/Screenshot 2024-10-26 at 7.50.18 PM.png)

Intuitively, this makes sense as loans for larger amounts will typically be paid off over a longer period of time. Some common examples would be 1-6 months for a layaway loan compared to a 6 year term for for automobile loan. In order to make sure that this correlation does not negatively affect our models, we can combine these into a single feature:

```
monthly_payment_amount = credit_amount / duration
```

We additionally create a feature to capture differences between short term loans and long term loans. [This resource released by the Corporate Finance Institute](https://corporatefinanceinstitute.com/resources/accounting/short-term-loan/) indicates that a short term loan is typically 6 to 18 months. The maximum `duration` in our data set is 72 so we create a new feature to indiciate whether or not if this can be considered a short term loan

```
is_short_term = duration > 18 ? 'no' : 'yes'
```

Upon closer inspection, some of these numeric features, namely `installment_commitment`, `residence_since`, `existing_credits`, and `num_dependents`, have a low degree of freedom (DOF), so we recategorize these are ordinal features.

### Transforming skewed features

The distribution of `monthly_payment_amount` and `age` are visibly skewed right so let's apply a log transformation on them to make them them closer to a normal distribution.

<img src="/Users/chanlee/Library/Application Support/typora-user-images/Screenshot 2024-10-26 at 8.43.22 PM.png" alt="Screenshot 2024-10-26 at 8.43.22 PM" style="zoom:50%;" />

This is strictly necessary for model building, particularly with Random Forest and KNN models, but it can still be useful when analying individual features with statical tests such as the ones we perform below

#### Analysis of individual features

I ran a Chi-square test on the categorical features to get an idea of what which features impact credit risk. Specifically, a chi-square test determines if two categorical variables (in this case, a feature and the response variable `class`) are dependent, i.e. if the feature influences the response.

<img src="/Users/chanlee/Library/Application Support/typora-user-images/Screenshot 2024-10-26 at 8.50.13 PM.png" alt="Screenshot 2024-10-26 at 8.50.13 PM" style="zoom:50%;" />

It appears that features such as `num_dependents` and `residence_since` have little to no impact on credit risk. Interestingly, `existing_credits` (the number of existing credits at this bank) also has minimal impact, something I did not expect.

For numeric features, I ran a t-test which has a similar premise to the chi-square test but we check if the mean of a variable between two groups, in this case `bad` and `good`, are statistically significant.

![Screenshot 2024-10-26 at 9.00.07 PM](/Users/chanlee/Library/Application Support/typora-user-images/Screenshot 2024-10-26 at 9.00.07 PM.png)

It appears both of these have significant impact on credit risk. Before running our models, it may also be worth "binning" age into several buckets to see if we can get a better interpretation.

![Screenshot 2024-10-26 at 9.11.58 PM](/Users/chanlee/Library/Application Support/typora-user-images/Screenshot 2024-10-26 at 9.12.30 PM.png)

Running the chi-square test on this now yields a comparable p-value of 0.003. We should consider using this instead of the raw `age` value since this will provide a more digestible interpretation.
