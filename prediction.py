'''
Predict Users' new booking by combining of RnadomForest and XGB classifiers.
'''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
from data_preparation import TRAINING_FINAL_CSV_FILE
from data_preparation import LABELS_FINAL_CSV_FILE
from data_preparation import TESTING_FINAL_CSV_FILE
from data_preparation import ACCOUNT_DATE_YEAR


# pylint: disable=fixme, no-member


DEPTH_XGB, ESTIMATORS_XGB, LEARNING_XGB, SUBSAMPLE_XGB, COLSAMPLE_XGB = (
    7, 60, 0.2, 0.7, 0.6)                # XGBoost parameters.

ESTIMATORS_RF, CRITERION_RF, DEPTH_RF, MIN_LEAF_RF, JOBS_RF = (
    400, 'gini', 20, 8, 30)              # RandomForestClassifier parameters.
FRESH_DATA_YEAR = 2014                   # Year when data is considered fresh.
SUBMISSION_CSV = 'final_prediction.csv'  # Where to store the predictions.


def perform_prediction_xgb_rf(training, labels, testing, xgb_votes, rf_votes):
    """ Perform prediction using a combination of XGB and RandomForests. """
    predictions = np.zeros((len(testing), len(set(labels))))
    # Predictions using xgboost.
    for i in range(xgb_votes):
        print 'XGB vote %d' % i
        xgb = XGBClassifier(
            max_depth=DEPTH_XGB, learning_rate=LEARNING_XGB,
            n_estimators=ESTIMATORS_XGB, objective='multi:softprob',
            subsample=SUBSAMPLE_XGB, colsample_bytree=COLSAMPLE_XGB)
        xgb.fit(training, labels)
        predictions += xgb.predict_proba(testing)
    # Predictions using RandomForestClassifier.
    for i in range(rf_votes):
        print 'RandomForest vote %d' % i
        rand_forest = RandomForestClassifier(
            n_estimators=ESTIMATORS_RF, criterion=CRITERION_RF, n_jobs=JOBS_RF, 
            max_depth=DEPTH_RF, min_samples_leaf=MIN_LEAF_RF, bootstrap=True)
        rand_forest.fit(training, labels)
        predictions += rand_forest.predict_proba(testing)
    return predictions


def main():
    """ Perform prediction. """
    training = pd.read_csv(TRAINING_FINAL_CSV_FILE, index_col=0)
    labels = pd.read_csv(LABELS_FINAL_CSV_FILE, index_col=0)
    testing = pd.read_csv(TESTING_FINAL_CSV_FILE, index_col=0)
    encoder = LabelEncoder()
    predictions = np.zeros((len(testing), len(set(labels))))

    # Use the full data set for the prediction.
    predictions += perform_prediction_xgb_rf(
        training, encoder.fit_transform(labels), testing, 5, 2)

    # Use only "fresh" data for prediction. Fresh data, are considered those
    # that are an ACCOUNT_DATE_YEAR equal or higher than FRESH_DATA_YEAR.

    training_fresh = training[training[ACCOUNT_DATE_YEAR] >= FRESH_DATA_YEAR]
    labels_fresh = labels.ix[training_fresh.index]
    predictions += perform_prediction_xgb_rf(
        training_fresh, encoder.fit_transform(labels_fresh), testing, 10, 4)

    # Use the 5 classes with highest scores.
    ids, countries = ([], [])
    for i in range(len(testing)):
        idx = testing.index[i]
        ids += [idx] * 5
        countries += encoder.inverse_transform(
            np.argsort(predictions[i])[::-1])[:5].tolist()

    # Save prediction in CSV file.
    sub = pd.DataFrame(
        np.column_stack((ids, countries)), columns=['id', 'country'])
    sub.to_csv(SUBMISSION_CSV, index=False)


if __name__ == '__main__':
    main()  # Run the main method.
