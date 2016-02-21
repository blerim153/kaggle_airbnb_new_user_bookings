		Airbnb New User Booking Competition (Kaggle)

Competition:
-------------------------------
https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings


Where to find the original data:
-------------------------------
https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data


Short description:
-------------------------------
An important characteristic of this competition is that there is a time cut-off
between training and testing data -- training data end on 7/1/2014, and testing
data start on that data. Moreover, the session informations are available only 
for the data points after 2014. 
This code used 4 classifiers: (1) XGB trained on all (training) data, (2)
RandomForests trained on all data, (3) XGB classifier trained on recent 
(aka fresh) data only, and  (4) RandomForests trained on all data only.
The results from each clasifier formed the final prediction via weighted voting. 


How to run the code:
-------------------------------
(0) Make sure that the raw data from the Kaggle competition are in the
    same directory as the code.

(1) Run the “feature_extraction_sessions.py” module:
    This will extract frequency counts from the “sessions.csv” data.
    The extracted features will be saved in the code directory. 

(2) Run the “data_preparation.py” module:
    This will make the data ready for the classifiers.
    The data processing includes:
        — Dealing with missing data.
        - Applying one-hot-encoding to categorical features.
        - Parsing dates (and extraction of simple numerical features).
        - Adding to the training and testing data the features that
          were extracted from sessions.csv file in the previous step.

(3) Run the prediction.py module:
    This will use the output of the previous step, to do the predictions.
    The final predictions will be saved in a file called 
    ’final_prediction.csv' 


Status:
-------------------------------
Finished in the 60th position in the (private) leaderboard (out
of 1463 participants).
