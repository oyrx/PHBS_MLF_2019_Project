## 2.1 correlation & feature engineering

Next, we will further process the data；aside from checking each attribute looking at the plots, is there a mathematical way to select the parameters that are strongly correlated with each other? The issue here is that reservation_status are linked with some categorical attributes so we cannot simply use "df.corr()" to accurately get the correlation matrix. 

### 1) Cramer's V model

*Cramer's V model* based on the chi squared satistic that can show how strongly nominal variables are associated with one another. This is very similar to correlation coefficient where 0 means no linear correlation and 1 means strong linear correlation.

**Drop some features**: As we did before, two features ("reservation_status_date" & "reservation_status") are dropped for avoidance of leakage. In addition, we drop the feature "arrival_date_year" because we will use future information to predict future cancellation behavior. 

**Results**: "deposit_type"  showed the highest correlation with the target variable. The reservation_status_date effect was already looked at in the previous section where we saw an intersting trend that people cancel less during the winter time.

### 2) numerical features's correlations

**Drop some features**: re-convert "is_canceled" attribute to numerical values. 

**Results**: both lead_time and total_of_special_requests had the strongest linear correlations with is_canceled target variable.

### 3) PCA analysis on categorical features 

**OneHotEncoding**:  To convert categorical features to numerical ones using Scikit-learn. This requires running integer encoding first follwed by OneHotEncoding.
Then we Running labelencoder and onehotencoder to convert to numerical features.
 
**Results**: the principal component 1 holds 44.2% of the information while the principal component 2 holds only 32.9% of the information. Summing them up, we will have ~77% of information.we need about 8 components to represent 90% of the dataset.

