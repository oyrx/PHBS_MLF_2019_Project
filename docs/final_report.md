# Cancel or not? Predictive Analysis for Hotel Booking Data

## 1 Background

A large number of reservation orders are generated every day on the hotel reservation website, and there are significant differences in the reservation time, reservation location, and number of customers for different orders.

In a large number of orders generated every day, after a part of the order is completed, the end user will not finalize the order and will cancel the order in the middle. This research is mainly to predict the behavior of canceling orders.

**Research purpose**: Through the multi-dimensional information of the order, predict whether different orders will be cancelled.

**Research significance**: Before the user cancels the order, it is predicted whether the user will cancel the order, which is beneficial to the hotel and the reservation website to better allocate resources, improve the true utilization rate of resources, and maximize the revenue. The problem of overselling airline tickets by analog airlines, overselling airline tickets within a reasonable range, helps to achieve a balance between customer efficiency and company revenue, and achieves the most profit without harming the customer experience.

## 2 Data description & cleaning

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

The prediction target of this study is the **is_canceled** indicator (0-1 variable), the data set contains a total of 31 dimensions of information. Among them, all 30 dimensions are discrete variables, and only **adr** (Average Daily Rate) is a continuous variable.
Because the data set owner has done preliminary data cleaning work, the data set quality is high. After statistics, it is found that there is no need to do too much data cleaning work, only a small amount of vacant values need to be filled. The data cleaning work done this time mainly includes:

- Fill the na value of the children factor. Considering that the children and babies factor have a small difference and the vacancy values of the children field are very few, they are filled directly with the babie field

- The other fields with vacant values are all categorical fields. Here we want to retain as many features as possible, so fill in the vacant values as **'undefined'** and do not delete them.
- 

## 3 Overall process

1. **Determine the data set**. There are a lot of open source data on kaggle, considering the significance of the topic and the difficulty of prediction, and finally select the hotel prediction topic.

2. **Identify the problem**. Taking into account the hotel booking time, whether to cancel the order is of great significance to the hotel or the user. At the same time, the cancellation of the predetermined behavior and other variables in the data set have a causal relationship, so we determine the research direction to determine whether the user cancels the order,
3. **Data exploration**. Correlation analysis of data and feature engineering processing of data sets using pca method.
4. **Classical model (tree based)**. In this section, initially, we'll start with two classical baseline models, logistic regression and randomforest with default parameter, and then use boosting techniques to improve the performance of tree-based models in two efficient modern frameworks, LightGBM and XGBoost.
5. **Deep learning model** (Simple ANN).
6. **Conclusion**
