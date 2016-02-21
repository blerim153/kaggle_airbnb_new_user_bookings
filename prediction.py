'''
Predict Users' new booking by combining of RnadomForest and XGB classifiers.
'''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
from data_preparation import TRAINING_FINAL_CSV_FILE  # File with train data.
from data_preparation import LABELS_FINAL_CSV_FILE    # File with labels.
from data_preparation import TESTING_FINAL_CSV_FILE   # File with test data.
from data_preparation import ACCOUNT_DATE_YEAR        # Where to find year.
from data_preparation import LABEL                    # How the label is called.


# pylint: disable=fixme, no-member


DEPTH_XGB, ESTIMATORS_XGB, LEARNING_XGB, SUBSAMPLE_XGB, COLSAMPLE_XGB = (
    7, 60, 0.2, 0.7, 0.6)                # XGBoost parameters.

ESTIMATORS_RF, CRITERION_RF, DEPTH_RF, MIN_LEAF_RF, JOBS_RF = (
    500, 'gini', 20, 8, 30)              # RandomForestClassifier parameters.
FRESH_DATA_YEAR = 2014                   # Year when data is considered fresh.
SUBMISSION_CSV = 'final_prediction.csv'  # Where to store the predictions.

# Tunning ensemble members. The votes show the importnce of each classfier
# in the final prediction.

XGB_ALL_VOTE, RF_ALL_VOTE, XGB_FRESH_VOTE, RF_FRESH_VOTE = (5, 2, 10, 4)


def perform_prediction(training, labels, testing, xgb_votes, rf_votes):
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
    train_df = pd.read_csv(TRAINING_FINAL_CSV_FILE, index_col=0)
    labels_df = pd.read_csv(LABELS_FINAL_CSV_FILE, index_col=0)
    test_df = pd.read_csv(TESTING_FINAL_CSV_FILE, index_col=0)
    assert set(train_df.index) == set(labels_df.index)

    encoder = LabelEncoder()
    encoder.fit(labels_df[LABEL])
    predictions = np.zeros((len(test_df), len(encoder.classes_)))

    # Use the full data set for the prediction.
    labels = encoder.transform(labels_df[LABEL])
    predictions += perform_prediction(
        train_df, labels, test_df, XGB_ALL_VOTE, RF_ALL_VOTE)

    # Use only "fresh" data for prediction. Fresh data, are considered those
    # that are an ACCOUNT_DATE_YEAR equal or higher than FRESH_DATA_YEAR.

    train_fresh = train_df[train_df[ACCOUNT_DATE_YEAR] >= FRESH_DATA_YEAR]
    labels_fresh = encoder.transform(labels_df.ix[train_fresh.index][LABEL])
    predictions += perform_prediction(
        train_fresh, labels_fresh, test_df, XGB_FRESH_VOTE, RF_FRESH_VOTE)

    # Use the 5 classes with highest scores.
    ids, countries = ([], [])
    for i in range(len(test_df)):
        idx = test_df.index[i]
        ids += [idx] * 5
        countries += encoder.inverse_transform(
            np.argsort(predictions[i])[::-1])[:5].tolist()

    # Save prediction in CSV file.
    sub = pd.DataFrame(
        np.column_stack((ids, countries)), columns=['id', 'country'])
    sub.to_csv(SUBMISSION_CSV, index=False)


if __name__ == '__main__':
    main()  # Run the main method.
