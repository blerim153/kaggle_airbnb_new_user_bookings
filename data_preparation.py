'''
Preparing the data for the classifiers.
'''
import datetime as dt
import numpy as np
import pandas as pd
from utilities import remove_rare_values_inplace


# pylint: disable=fixme, no-member


LABEL = 'country_destination'
CATEGORICAL_FEATURES = ['affiliate_channel', 'affiliate_provider',
                        'first_affiliate_tracked', 'first_browser',
                        'first_device_type', 'gender', 'language', 'signup_app',
                        'signup_method', 'signup_flow']

DATE_FORMAT = '%Y-%m-%d'                # Expected format for date.
ACCOUNT_DATE = 'date_account_created'   # Date column that will be exploited.
ACCOUNT_DATE_YEAR = '%s_%s' % (ACCOUNT_DATE, 'year')
ACCOUNT_DATE_MONTH = '%s_%s' % (ACCOUNT_DATE, 'month')
UNUSED_DATE_COLUMNS = ['timestamp_first_active', 'date_first_booking']

TRAIN_DATA_BASIC = 'train_users.csv'
TEST_DATA_BASIC = 'test_users.csv'
SESSION_DATA = 'session_features.csv'
TRAINING_FINAL_CSV_FILE = 'training_features.csv'
TESTING_FINAL_CSV_FILE = 'testing_features.csv'
LABELS_FINAL_CSV_FILE = 'labels.csv'

# A parameter to speed-up computation. Categorical values that appear
# less than the threshold will be removed.
VALUE_THRESHOLD = 0.001


def _parse_date(date_str, format_str):
    """ Extract features from the data_account_creted column.

    Warning: There is strong dependency between this method and the method
    replace_dates_inplace.

    Args:
        date_str -- A string containing a date value.
        str_format -- The format of the string date.

    Returns:
        A list of 4 values containing the extracted [year, month, day, weekend].
    """
    time_dt = dt.datetime.strptime(date_str, format_str)
    return [time_dt.year, time_dt.month, time_dt.day, time_dt.weekday()]


def extract_dates_inplace(features, date_column):
    """ Extract from the date-columns, year, month, and other numericals.

    Warning: There is strong dependency between this method and _parse_date.
    """
    extracted_vals = np.vstack(features[date_column].apply(
        (lambda x: _parse_date(x, DATE_FORMAT))))
    for i, period in enumerate(['year', 'month', 'day', 'weekday']):
        features['%s_%s' % (date_column, period)] = extracted_vals[:, i]
    features.drop(date_column, inplace=True, axis=1)


def apply_one_hot_encoding(pd_frame, column_list):
    """ Apply One-Hot-Encoding to pd_frame's categorical columns.

    Args:
        df_frame -- A pandas data frame.
        column_list -- A list of categorical columns, in df_frame.

    Returns:
        A pandas dataframe where the colums in column_list have been replaced
            by one-hot-encoded-columns.
    """
    new_column_list = []
    for col in column_list:
        tmp = pd.get_dummies(pd_frame[col], prefix=col)
        new_column_list.append(tmp)
    new_pd_frame = pd.concat(new_column_list+[pd_frame], axis=1)
    new_pd_frame.drop(column_list, inplace=True, axis=1)
    return new_pd_frame


def get_basic_train_test_data():
    """ Load the basic data in a pandas dataframe, and pre-process them. """
    training = pd.read_csv(TRAIN_DATA_BASIC, index_col=0)
    testing = pd.read_csv(TEST_DATA_BASIC, index_col=0)
    labels = training[LABEL].copy()
    training.drop(LABEL, inplace=True, axis=1)
    features = pd.concat((training, testing), axis=0)
    features.fillna(-1, inplace=True)

    # Process all features by removing rare values, appling one-hot-encoding to
    # those that are categorical and extracting numericals from ACCOUNT_DATE.

    remove_rare_values_inplace(features, CATEGORICAL_FEATURES, VALUE_THRESHOLD)
    features = apply_one_hot_encoding(features, CATEGORICAL_FEATURES)
    extract_dates_inplace(features, ACCOUNT_DATE)
    features.drop(UNUSED_DATE_COLUMNS, inplace=True, axis=1)
    return features, labels, training.index, testing.index


def main():
    """ Load basic data, add session data, and prepare them for predition. """
    features, labels, training_ids, testing_ids = get_basic_train_test_data()
    sessions = pd.read_csv(SESSION_DATA, index_col=0)
    features = pd.concat((features, sessions), axis=1)
    features.fillna(-1, inplace=True)
    # Save data training and testing data.
    training = features.ix[training_ids]
    testing = features.ix[testing_ids]

    # Warning: When saving the data, it's important that the header is True,
    # because labels is of type pandas.core.series.Series, while training is of
    # type pandas.core.frame.DataFrame, and they have different default values
    # for the header argument.

    assert set(training.index) == set(labels.index)
    training.to_csv(TRAINING_FINAL_CSV_FILE, header=True)
    testing.to_csv(TESTING_FINAL_CSV_FILE, header=True)
    labels.to_csv(LABELS_FINAL_CSV_FILE, header=True)


if __name__ == '__main__':
    main()  # Run the main method.
