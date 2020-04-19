# Cancel or not? Predictive Analysis for Hotel Booking Data

(PHBS_MLF_2019_Project / [Course Link](https://github.com/PHBS/MLF))

## Members

| Group 13                                        | SID        |
| ----------------------------------------------- | ---------- |
| [Jingwei Gao](https://github.com/LobbyBoy-Dray) | 1801213126 |
| [Simon SHEN](https://github.com/Simon9511)      | 1801212832 |
| [Yingjie Jiang](https://github.com/Jason422)    | 1901212596 |
| [Rongxin Ouyang](https://github.com/oyrx)       | 1801213136 |

## Project Description

| Project              | Details                                                                                                                                                                                                                                                                    |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Goal**             | To predict whether a specific hotel booking will be cancelled.                                                                                                                                                                                                             |
| **Data**             | [Hotel booking demand](https://www.kaggle.com/jessemostipak/hotel-booking-demand) on Kaggle                                                                                                                                                                                |
| **Data File**        | hotel_bookings.csv                                                                                                                                                                                                                                                         |
| **Data Description** | This data set contains booking information for a city hotel and a resort hotel, and includes information such as when the booking was made, length of stay, the number of adults, children, and/or babies, and the number of available parking spaces, among other things. |

<details>
<summary>Target variable</summary>

- `is_canceled`: Value indicating if the booking was canceled (1) or not (0)
</details>
<details>
<summary>Features Description</summary>

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

</details>

## Models

- Logistic Regression
- Random Forest
- Dart in LightGBM
- GBDT in XGBoost
- ANN

## Result

### [Final Report](./docs/Final_report.md)

Accuracy of all models:
| Accuracy | Logistic Regression | Random Forest | LightGBM (DART) | XGBoost (GBDT) | ANN |
| ------------------- | ------------------- | ------------- | --------------- | -------------- | ----- |
| **Test** | 0.79433 | 0.8925 | 0.89614 | 0.89404 | 0.875 |

- [Baseline and Tree-based Models](./docs/Baseline_and_tree_models.md)  
  Accuracy of baseline and tree-based models:

  | Metrics              | Logistic Regression | Random Forest | LightGBM (DART) | XGBoost (GBDT) |
  | -------------------- | ------------------- | ------------- | --------------- | -------------- |
  | **Accuracy (Train)** | 0.79377             | 0.97796       | 0.98896         | 0.99574        |
  | **Accuracy (Test)**  | 0.79433             | 0.8925        | 0.89614         | 0.89404        |
  | **Precision**        | 0.80                | 0.89          | 0.90            | 0.89           |
  | **Recall**           | 0.79                | 0.89          | 0.90            | 0.89           |
  | **F1-score**         | 0.79                | 0.89          | 0.90            | 0.89           |

- [ANN](./docs/ANN.md)  
   <img src="./images/retrain.png" width="600" align="center">

### [Pre-trained Models](https://github.com/oyrx/PHBS_MLF_2019_Project_Models)

Dumped models for download.  
<img src="./images/LighGBM_small.png" width="600" align="center">

## Repo Structure

\- **code\\** _All the Jupyter Notebooks_  
\- \- **Exploration\*.ipynb** _Prepocessing and EDA_  
\- \- **Modelling\*.ipynb** _Modelling, LightGBM, XGBoost,and Neural Networks_  
\- **data\\** _Data_  
\- \- **feature_description.csv** _Description of each feature_  
\- \- **hotel_booking_cleaned.csv** _Cleand full dataset_  
\- \- **hotel_booking.csv** _Original full dataset_  
\- \- **Train.csv** _Data for training_  
\- \- **Test.csv** _Data for testing_  
\- **docs\\** _Project documentation_  
\- \- **Descriptive_Report.html** _Generated by pandas_profiling_  
\- **images\\** _image resources_  
\- **[models](https://github.com/oyrx/PHBS_MLF_2019_Project_Models)\\** _Pre-trained Models in a seperated repo._  
\- \- \.\.\.  
\- **Report.pdf** _Final report_

## Aknowledgements

- The data is originally from the article [Hotel Booking Demand Datasets](https://www.sciencedirect.com/science/article/pii/S2352340918315191), written by Nuno Antonio, Ana Almeida, and Luis Nunes for Data in Brief, Volume 22, February 2019.
- The data was downloaded and cleaned by Thomas Mock and Antoine Bichat for [#TidyTuesday during the week of February 11th, 2020.](https://github.com/rfordatascience/tidytuesday/blob/master/data/2020/2020-02-11/readme.md)

## References

- Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. Data in brief, 22, 41-49.
