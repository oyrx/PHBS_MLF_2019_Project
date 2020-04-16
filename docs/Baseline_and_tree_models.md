## 3. Baseline and tree-based models

### 1) Logistic Regression (baseline_1)

Default parameter with `l2` regularization for comparison.

### 2) Random Forest (baseline_2)

Default parameter with limited `n_estimators` for comparison.

### 3) GBDT and DART in LightGBM

#### What is GBDT and DART?

Gradient Boosted Decision Trees (GBDT) is a machine learning algorithm that iteratively constructs an ensemble of weak decision tree learners through boosting.

#### Why GBDT or DART?

For GBDT:

- Feature selection is inherently performed during the learning process
- Not prone to collinear/identical features
- Models are relatively easy to interpret
- Easy to specify different loss functions

For DART:

- DART [may be more accurate than GBDT](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#deal-with-over-fitting)

#### Why LightGBM?

#### Preprocessing

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

#### Parameter Tuning

The process of tuning optimized parameter typically complies with latest [manuscript in official docementation](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#deal-with-over-fitting).

First, experiments were conducted to find a generally optimized parameter dict of `num_leaves`, `min_data_in_leaf` and `max_depth`.

Second, tuning other paramters to get higher accuracy in both training data and testing data, where slightly over-fitting on testing set is accpetable.

Then, apply [regularization and other constraints](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#deal-with-over-fitting) to tackle over-fitting.

In order to improve computational performance, sub-sampling and limited cross validation folds are consecutively applied in the whole process.

<div align="center"><img src="../models/LightGBM_04161347_cvresult.png"></div>

\* _Parameters ange were selected on previous training results and not continouous._

### 4) XGBoost

#### What is XGBoost?

XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.

#### Why XGBoost?

It provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way.

<div align="center"><img src="../images/tree_growth.jpg"></div>

We're curious about the diffence between [level-wise tree growth and leaf-wise tree growth](https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/), thus, we decide to run both LightGBM and XGBoost.

Implementation and tuning are similar to LightGBM though caterical features in numeric way is acceptive in XGBoost.

### Result

- A LightGBM Model with accuracy upto approximately 90%()
- Manifold predominant features: `lead_time`, `adr`(Average Daily Rate, dividing the sum of all lodging transactions by the total number of staying nights), `arrival_date_day_of_month`, `arrival_date_week_number`, `country`, `agent`, etc., representing characteristics of time, place, actor, laws of normal transactions, and so on.
- Relatively easy-to-interpretant tree model

### Results in Detail

**Comparison Among Models**
| | Logistic Regression | Random Forest | LightGBM (DART) | XGBoost (GBDT) |
|---|---|---|---|---|
| **Accuracy (Train)** | 0.79377 | 0.97796| 0.98896 |0.99574 |
| **Accuracy (Test)** | 0.79433 |0.8925 | 0.89614 |0.89404 |
| **Precision** | 0.80 |0.89 |0.90 |0.89 |
| **Recall** |0.79 |0.89 |0.90 |0.89 |
| **F1-score** | 0.79 |0.89 | 0.90|0.89 |
| **Parameters** | `{'boosting': 'DART', 'feature_fraction': 0.7, 'lambda_l2': 0.1, 'max_depth': 25, 'min_split_gain': 0.1, 'n_estimators': 3000, 'num_leaves': 100, 'objective': 'binary'}` | `{'n_estimators': '100', 'max_depth': 25, 'random_state' : 0, 'bootstrap': True}` | `{'boosting': 'DART', 'feature_fraction': 0.7, 'lambda_l2': 0.1, 'max_depth': 25, 'min_split_gain': 0.1, 'n_estimators': 3000, 'num_leaves': 100, 'objective': 'binary'}` | `{'colsample_bytree': 0.7, 'max_depth': 50, 'n_estimators': 100, 'reg_alpha': 1.3, 'reg_lambda': 1.1, 'subsample': 0.9}`| \* _Abbr. for Gradient Boosted Decision Trees_  
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

<div align="center"><img src="../models/LightGBM_04161347.png"></div>

**Tree Based Model Plot of Best Model**  
![TreeLightGBM](../images/LighGBM_small.png)  
\* See: [full tree](https://github.com/oyrx/PHBS_MLF_2019_Project/raw/master/images/LightGBM_small.png)

**Feature importance in Best Model**

<div align="center"><img src="../images/LightGBM_feature_importance.jpg"></div>
