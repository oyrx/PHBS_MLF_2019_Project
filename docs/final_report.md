# Cancel or not? Predictive Analysis for Hotel Booking Data

<!-- TOC -->

- [Cancel or not? Predictive Analysis for Hotel Booking Data](#cancel-or-not-predictive-analysis-for-hotel-booking-data)
  - [Task and Significance](#task-and-significance)
  - [Data Exploration](#data-exploration)
    - [Feature with meaning](#feature-with-meaning)
    - [Correlations and PCA](#correlations-and-pca)
      - [Cramer's V model](#cramers-v-model)
      - [Numerical features's correlations](#numerical-featuress-correlations)
      - [PCA analysis on categorical features](#pca-analysis-on-categorical-features)
  - [Data Cleaning](#data-cleaning)
  - [Baseline and tree-based models](#baseline-and-tree-based-models)
    - [Methodology](#methodology)
      - [What is GBDT and DART?](#what-is-gbdt-and-dart)
      - [Why LightGBM and XGBoost?](#why-lightgbm-and-xgboost)
    - [Preparation](#preparation)
    - [Models](#models)
      - [Baseline: Logistic Regression and Random Forest](#baseline-logistic-regression-and-random-forest)
      - [DART in LightGBM and GBDT in XGBoost](#dart-in-lightgbm-and-gbdt-in-xgboost)
    - [Result](#result)
      - [Results in short](#results-in-short)
      - [Results in detail](#results-in-detail)
  - [Deep learning model](#deep-learning-model)
    - [Toolkit: PyTorch](#toolkit-pytorch)
    - [Data preprocessing](#data-preprocessing)
    - [Network structure](#network-structure)
    - [Hyperparameter tuning](#hyperparameter-tuning)
    - [Retrain & Test](#retrain--test)
    - [Summary](#summary)
  - [Conclusion](#conclusion)

<!-- /TOC -->

Emerging network society issues new challenges on understanding big data in electronic consuming behaviors, as well as in hotel business. Significant differences can be easily found in tremendous reservation records, consisting of time, location, historic characteristics, and so on.

Considering oblivious risks by potential cancellation after reservation, the utilization of hotel booking data can be conducive to optimizing business decisions and strategies, nevertheless, also far from application without quantified nuances behind the topsoils.

## Task and Significance

**Project Task**  
This project is proposed to use order information of multiple dimensions order to predict whether a specific order will be cancelled or not.

**Significance**  
Before the user cancels the order, it is predicted whether the user will cancel the order, which is beneficial to the hotel and the reservation website to better allocate resources, improve the true utilization rate of resources, and maximize the revenue. The problem of overselling airline tickets by analog airlines, overselling airline tickets within a reasonable range, helps to achieve a balance between customer efficiency and company revenue, and achieves the most profit without harming the customer experience.

**Process**

1. **Determine the data set**. There are a lot of open source data on kaggle, considering the significance of the topic and the difficulty of prediction, and finally select the hotel prediction topic.

2. **Identify the problem**. Taking into account the hotel booking time, whether to cancel the order is of great significance to the hotel or the user. At the same time, the cancellation of the predetermined behavior and other variables in the data set have a causal relationship, so we determine the research direction to determine whether the user cancels the order,
3. **Data exploration**. Correlation analysis of data and feature engineering processing of data sets using pca method.
4. **Classical model (tree based)**. In this section, initially, we'll start with two classical baseline models, logistic regression and randomforest with default parameter, and then use boosting techniques to improve the performance of tree-based models in two efficient modern frameworks, LightGBM and XGBoost.
5. **Deep learning model**. Although we have derived a nice result (i.e. high accuracy) from gradient boosting algorithms, we still want to know how the deep learning model performs in this task. We choose to use a simple feed-forward neural network as our deep learning model. We fix the network structure in advance and do some hyperparameter tuning to find whether it is possible to get a better result.

## Data Exploration

Incongruous exploration on the meanings of features was conducted before formal exploratory analysis.

### Feature with meaning

- `hotelHotel`: (H1 = Resort Hotel or H2 = City Hotel)
- `lead_time`: Number of days that elapsed between the entering date of the booking into the PMS and the arrival date
- `arrival_date_year`: Year of arrival date
- `arrival_date_month`: Month of arrival date
- `arrival_date_week_number`: Week number of year for arrival date
- `arrival_date_day_of_month`: Day of arrival date
- `stays_in_weekend_nights`: Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel
- `stays_in_week_nights`: Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel
- `adults`: Number of adults
- `children`: Number of children
- `babies`: Number of babies
- `meal`: Type of meal booked. Categories are presented in standard hospitality meal packages: Undefined/SC – no meal package; BB – Bed & Breakfast; HB – Half board (breakfast and one other meal – usually dinner); FB – Full board (breakfast, lunch and dinner)
- `country`: Country of origin. Categories are represented in the ISO 3155–3:2013 format
- `market_segment`: Market segment designation. In categories, the term “TA” means “Travel Agents” and “TO” means “Tour Operators”
- `distribution_channel`: Booking distribution channel. The term “TA” means “Travel Agents” and “TO” means “Tour Operators”
- `is_repeated_guest`: Value indicating if the booking name was from a repeated guest (1) or not (0)
- `previous_cancellations`: Number of previous bookings that were cancelled by the customer prior to the current booking
- `previous_bookings_not_canceled`: Number of previous bookings not cancelled by the customer prior to the current booking
- `reserved_room_type`: Code of room type reserved. Code is presented instead of designation for anonymity reasons.
- `assigned_room_type`: Code for the type of room assigned to the booking. Sometimes the assigned room type differs from the reserved room type due to hotel operation reasons (e.g. overbooking) or by customer request. Code is presented instead of designation for anonymity reasons.
- `booking_changes`: Number of changes/amendments made to the booking from the moment the booking was entered on the PMS until the moment of check-in or cancellation
- `deposit_type`: Indication on if the customer made a deposit to guarantee the booking. This variable can assume three categories: No Deposit – no deposit was made; Non Refund – a deposit was made in the value of the total stay cost; Refundable – a deposit was made with a value under the total cost of stay.
- `agent`: ID of the travel agency that made the booking
- `company`: ID of the company/entity that made the booking or responsible for paying the booking. ID is presented instead of designation for anonymity reasons
- `days_in_waiting_list`: Number of days the booking was in the waiting list before it was confirmed to the customer
- `customer_type`: Type of booking, assuming one of four categories: Contract - when the booking has an allotment or other type of contract associated to it; Group – when the booking is associated to a group; Transient – when the booking is not part of a group or contract, and is not associated to other transient booking; Transient-party – when the booking is transient, but is associated to at least other transient booking
- `adr`: Average Daily Rate as defined by dividing the sum of all lodging transactions by the total number of staying nights
- `required_car_parking_spaces`: Number of car parking spaces required by the customer
- `total_of_special_requests`: Number of special requests made by the customer (e.g. twin bed or high floor)
- `reservation_status`: Reservation last status, assuming one of three categories: Canceled – booking was canceled by the customer; Check-Out – customer has checked in but already departed; No-Show – customer did not check-in and did inform the hotel of the reason why
- `reservation_status_date`: Date at which the last status was set. This variable can be used in conjunction with the ReservationStatus to understand when was the booking canceled or when did the customer checked-out of the hotel

### Correlations and PCA

With the help of our [manual work](https://github.com/oyrx/PHBS_MLF_2019_Project/blob/master/code/Corrleations_And_Feature_Engineering.ipynb) and [pandas_profiling](https://github.com/oyrx/PHBS_MLF_2019_Project/blob/master/code/Exploration_Statistics.ipynb), we discern that:

#### Cramer's V model

_Cramer's V model_ based on the chi squared satistic that can show how strongly nominal variables are associated with one another. This is very similar to correlation coefficient where 0 means no linear correlation and 1 means strong linear correlation.

**Drop some features**: As we did before, two features ("reservation_status_date" & "reservation_status") are dropped for avoidance of leakage. In addition, we drop the feature "arrival_date_year" because we will use future information to predict future cancellation behavior.

**Results**: "deposit_type" showed the highest correlation with the target variable. The reservation_status_date effect was already looked at in the previous section where we saw an intersting trend that people cancel less during the winter time.

#### Numerical features's correlations

**Drop some features**: re-convert "is_canceled" attribute to numerical values.

**Results**: both lead_time and total_of_special_requests had the strongest linear correlations with is_canceled target variable.

#### PCA analysis on categorical features

**OneHotEncoding**: To convert categorical features to numerical ones using Scikit-learn. This requires running integer encoding first follwed by OneHotEncoding.
Then we Running labelencoder and onehotencoder to convert to numerical features.

**Results**: the principal component 1 holds 44.2% of the information while the principal component 2 holds only 32.9% of the information. Summing them up, we will have ~77% of information.we need about 8 components to represent 90% of the dataset.

Other details of each feature can be found at [descriptive report](https://github.com/oyrx/PHBS_MLF_2019_Project/blob/master/docs/Descriptive_Report.html)

## Data Cleaning

The prediction target of this study is the **is_canceled** indicator (0-1 variable), the data set contains a total of 31 dimensions of information. Among them, all 30 dimensions are discrete variables, and only **adr** (Average Daily Rate) is a continuous variable.
Because the data set owner has done preliminary data cleaning work, the data set quality is high. After statistics, it is found that there is no need to do too much data cleaning work, only a small amount of vacant values need to be filled. The data cleaning work done this time mainly includes:

- Fill the na value of the children factor. Considering that the children and babies factor have a small difference and the vacancy values of the children field are very few, they are filled directly with the babie field

- The other fields with vacant values are all categorical fields. Here we want to retain as many features as possible, so fill in the vacant values as **'undefined'** and do not delete them.

## Baseline and tree-based models

This section contains two baseline models, LR and Random Forest, and other two moder boosting methods, Dart in LightGBM and GBDT in XGBoost

### Methodology

#### What is GBDT and DART?

Gradient Boosted Decision Trees (GBDT) is a machine learning algorithm that iteratively constructs an ensemble of weak decision tree learners through boosting.

- For GBDT:

  - Feature selection is inherently performed during the learning process
  - Not prone to collinear/identical features
  - Models are relatively easy to interpret
  - Easy to specify different loss functions

- For DART:
  - Similar to GBDT but [may be more accurate than GBDT](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#deal-with-over-fitting)

#### Why LightGBM and XGBoost?

**Why do we deliberately use those two similar, to some extend, boosting framework?** The first reason is that DART, a slightly different method, is also comprised in LightGBM, which would provide diversity for our potential model candidates. Second, XGBoost and LightGBM use discrepent tree growth strategies ( level-wise vs. leaf-wise) and the difference should not been ignored in finding the best hyper-parameters, especially when that level-wise leads to unexpected ramifications like over-fitting is literally a commonplace for professional data-scientists.

<div align="center"><img src="../images/tree_growth.jpg" width="400"></div>

We're curious about the nuances between [level-wise tree growth and leaf-wise tree growth](https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/), thus, we decide to run both LightGBM and XGBoost.

Implementation and tuning are similar to LightGBM though caterical features in numeric way is acceptable in XGBoost.

### Preparation

Based on previously cleaned and splitted datasets, consistent standarization and some extra process were carried out to fit model requirements.

Fairly significant issue here is datatype. According to the design and implementation of LightGBM, categorical features [should be kept in interger](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#categorical-feature-support), thereby, the process of standarization was divided into two different chunks to relinquish categorial features and then bring them back.

Incidentally, for engineering convinience, we also introduced a redesigned function named "algorithm_pipeline()" to expedite implementation through predefined datasets, fit criteria, and reusable grid search process.

```python
def algorithm_pipeline(model, \
                       param_grid, \
                       X_train_data = X_train_std, \
                       X_test_data = X_test_std, \
                       y_train_data = y_train, \
                       y_test_data = y_test, \
                       cv=10, \
                       scoring_fit='accuracy',
                       do_probabilities = False):

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring=scoring_fit,
        verbose=2,
        refit=True # return the best refitted model
    )

    fitted_model = gs.fit(X_train_data, y_train_data)

    if do_probabilities:
      y_pred = fitted_model.predict_proba(X_test_data)
    else:
      y_pred = fitted_model.predict(X_test_data)

    return fitted_model, y_pred
```

### Models

#### Baseline: Logistic Regression and Random Forest

Starting with two baseline models, a logistic regression with l2 regularization and a random forest model with limited n_estimators, we find that a simple logistic regression is literally “not bad” as an approximately 80% accuracy on the testing set and random forest performs much better, scoring 89%.

However, random forest is very slow for training even a single model even with this constrained n_estimators. Heeding the advice from Jingwei Gao, we decide to learn the applications of two modern boosting framework, XGBoost and LightGBM to accelerate parameter tuning process.

#### DART in LightGBM and GBDT in XGBoost

With the help of scaffolding, those two modules and one customized grid search function, manifold combinations of hyperparameters are efficient tested according to the manuscript in [official docementation](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#deal-with-over-fitting) in following process:

- First, experiments were conducted to find a generally optimized parameter dict of num_leaves, min_data_in_leaf and max_depth.

- Second, tuning other paramters to get higher accuracy in both training data and testing data, where slightly over-fitting on testing set is accpetable.

- Then, apply [regularization and other constraints](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#deal-with-over-fitting) and other constraints to tackle over-fitting.

- In order to improve computational performance, sub-sampling and limited cross validation folds are consecutively applied in the whole process.

<div align="center"><img src="https://github.com/oyrx/PHBS_MLF_2019_Project/raw/master/images/LightGBM_04161347_cvresult.png"></div>

\* _Parameters ange were selected on previous training results and not continouous due to limited computation capacity_

Models with boosting techniques take the crown with testing accuracy up to 89.61%(DART in LightGBN).

### Result

#### Results in short

- A DART(LightGBM) Model with accuracy upto approximately 90%(89.61%).
- Manifold predominant features: `lead_time`, `adr`(Average Daily Rate, dividing the sum of all lodging transactions by the total number of staying nights), `arrival_date_day_of_month`, `arrival_date_week_number`, `country`, `agent`, etc., representing characteristics of time, place, actor, laws of normal transactions, and so on.
- Relatively easy-to-interpretant tree model

#### Results in detail

**Comparison Among Models**
| | Logistic Regression | Random Forest | LightGBM (DART) | XGBoost (GBDT) |
|---|---|---|---|---|
| **Accuracy (Train)** | 0.79377 | 0.97796| 0.98896 |0.99574 |
| **Accuracy (Test)** | 0.79433 |0.8925 | 0.89614 |0.89404 |
| **Precision** | 0.80 |0.89 |0.90 |0.89 |
| **Recall** |0.79 |0.89 |0.90 |0.89 |
| **F1-score** | 0.79 |0.89 | 0.90|0.89 |
| **Parameters** | `{'boosting': 'DART', 'feature_fraction': 0.7, 'lambda_l2': 0.1, 'max_depth': 25, 'min_split_gain': 0.1, 'n_estimators': 3000, 'num_leaves': 100, 'objective': 'binary'}` | `{'n_estimators': '100', 'max_depth': 25, 'random_state' : 0, 'bootstrap': True}` | `{'boosting': 'DART', 'feature_fraction': 0.7, 'lambda_l2': 0.1, 'max_depth': 25, 'min_split_gain': 0.1, 'n_estimators': 3000, 'num_leaves': 100, 'objective': 'binary'}` | `{'colsample_bytree': 0.7, 'max_depth': 50, 'n_estimators': 100, 'reg_alpha': 1.3, 'reg_lambda': 1.1, 'subsample': 0.9}`|

\* _Abbr. for Gradient Boosted Decision Trees_  
\* _Small n_estimators in Random Forest on purpose._

**Metric Report and Confusion Matrix of Best Model**

Train(accuracy): 98.896%  
Test(accuracy): 89.614%
| Value | precision | recall | f1-score | support |
|---|---|---|---|---|
| 0 | 0.91 | 0.93 | 0.92 | 15033 |
| 1 | 0.88 | 0.84 | 0.86 | 8845 |
| **General** |
|accuracy|-|-|0.90|23878|
|macro avg|0.89|0.88|0.89|23878|
|weighted avg|0.90|0.90|0.90|23878|

<div align="center"><img src="https://github.com/oyrx/PHBS_MLF_2019_Project/raw/master/images/LightGBM_04161347.png"></div>

**Tree Based Model Plot of Best Model**  
![TreeLightGBM](../images/LighGBM_small.png)  
\* See: [full tree](https://github.com/oyrx/PHBS_MLF_2019_Project/raw/master/images/LightGBM_small.png)

**Feature importance in Best Model**

<div align="center"><img src="../images/LightGBM_feature_importance.jpg"></div>

## Deep learning model

> [Colab - Notebook - All codes](https://colab.research.google.com/drive/1TAiVwkV5Eh9kjxE1w9OeJEc1YPGsTrT_)

Although we have derived a beautiful result (i.e. high accuracy) from gradient boosting algorithms, we still want to know how the deep learning model performs in this task. Because there is little information about time series in this data set, we choose to use a simple **feed-forward neural network** as our deep learning model.

### Toolkit: PyTorch

<img src="https://pytorch.org/assets/images/pytorch-logo.png" width="100">

_PyTorch_ is an open source machine learning library and is widely used in deep learning scenarios for its flexibility. PyTorch uses dynamic computational graphs rather than static graphs, which can be regarded as the mean difference between it and other deep learning frameworks. To get more information on PyTorch, click [here](https://pytorch.org/).

### Data preprocessing

**Drop some features**: As we did before, two features ("reservation_status_date" & "reservation_status") are dropped for avoidance of leakage. In addition, we drop the feature "arrival_date_year" because we will use future information to predict future cancellation behavior.

**One-hot encoding**: One-hot encoder is used to convert categorical data into integer data. Since there are many categories under "company" and "agent", data's dimension increases to 1025 after the one-hot encoding.

**Validation set**: We use 20% data in `train.csv` as our validation data.

### Network structure

<img src="../images/structure.png" width="700" align="center">

- Input→1000→500→250→100→20→2
- **Dropout** after doing **batch normalization**
- Choose **Sigmoid/Tanh/ReLU** as activation function

### Hyperparameter tuning

It is a binary classification task, so we use **cross-entropy** as our loss function and apply **early stopping** to avoid over-fitting. Because we use dropout as a tool of regularization, we need to determine the **dropout rate** _dr_. We use Adam as adaptive learning rate method and fix _beta_1_ and _beta_2_ by using their default values, but we still need to determine the **learning rate** _lr_. At last, we want to compare the average performance of three kinds of **activation function** (sigmoid, Tanh, ReLU). Hence, there are three kinds of parameters that need to be tuned:

- learning rate _lr_ for Adam: _[0.005, 0.05]_
- dropout rate _dr_: _[0.2, 0.8]_
- activation function: sigmoid, tanh, ReLU

We use **random search** rather than grid search for hyperparameter tuning —— Randomly select **120** parameter combinations. The result is as follows:

<img src="../images/tune.png" width="600" align="center">

The best parameters among these 120 combinations are:

- learning rate: 0.00867688
- dropout rate: 0.260069
- activation function: ReLU

The corresponding **validation loss** is 0.283972. The **validation accuracy** is 0.870927. From the scatterplot, we can see that ReLU is a better choice for activation function because of its stableness. When dropout rate is high (0.6~0.8), using sigmoid or tanh as activation function will get bad results (loss approx 1.0). However, ReLU can still provide a small loss and high accuracy in that region.

### Retrain & Test

At last, we use the hyperparameters from the last step and retrain the model on the whole training data (original training set + validation set). The learning process:

<img src="../images/retrain.png" width="600" align="center">

The test loss is about 0.280 and test accuracy is about 0.875. Other performance metrics on test set:

<img src="../images/pmetrics.png" width="500" align="center">

### Summary

Results from multifold models, including traditional ML techniques and DL strategies, indicate **the failure of deep learning model** to defeat the gradient boosting methods.

This unanticipated ramification can be ascribed to the following reasons.

- **Misplaced advantages**  
  It's commonplace that a deep learning model is more efficient dealing with **unstructured data** such as images and text by extracting meaningful representations. However, all the data here is highly structured, which creates convenience for conventional models.

- **Parameter dilemma**  
  Deep learning models need to adjust **more parameters** in order to get better results in such context. Among all the hyperparameters, network structure is quite important, however, due to limited computation capacity **the network structure has to be fixed in advance**. That's to say, a trap of network structure has impeded our progress at the very beginning.

## Conclusion

TODO:
