'''
Extracting freatures from the session data.

The data set can be found here:
www.kaggle.com/c/airbnb-recruiting-new-user-bookings/download/sessions.csv.zip
'''
import collections
import numpy as np
import pandas as pd
import scipy.stats as stats


# pylint: disable=fixme, no-member


def remove_rare_values_inplace(df_frame, column_list, threshold):
    """ Remove rare values to speed up computation.

    Args:
        df_frame -- A pandas data frame.
        column_list -- A list of columns.
        threshold -- The threshold, below which a value is removed.
    """
    insignificant_population = int(np.floor(threshold * len(df_frame)))
    for cat in column_list:
        freqs = collections.Counter(df_frame[cat])
        other = [i for i in freqs if freqs[i] < insignificant_population]
        for i in other:
            df_frame[cat].replace(i, 'other', inplace=True)


def extract_frequency_counts(pd_frame, column_list):
    """ Extract frequency counts from pd_frame.

    For each index (that correspond to a user) this method will count the
    number of times that C == Ci, where C is a column in column_list, and Ci
    is a unique value of that column. The arg olumn_list is assumed
    to contain categorical columns.

    Args:
        df_frame -- A pandas data frame.
        column_list -- A list of columns.

    Returns:
        A pandas DataFrame, containing frequency counts.
    """
    df_extracted_sessions = []
    for col in column_list:
        for val in set(pd_frame[col]):
            print 'Frequency counts for (column: %s == value:%s)' % (col, val)
            tmp_df = pd_frame.groupby(pd_frame.index).apply(
                lambda group, x=col, y=val: np.sum(group[x] == y))
            tmp_df.name = '%s=%s' % (col, val)
            df_extracted_sessions.append(tmp_df)
    frequency_counts = pd.concat(df_extracted_sessions, axis=1)
    return frequency_counts


def extract_distribution_stats(pd_frame, numerical_col):
    """ Extract simple distribution statistics from a numerical column.

    Args:
        df_frame -- A pandas data frame.
        numerical_col -- A column in pd_frame that contains numerical values.

    Returns:
        A pandas DataFrame, containing simple satistics for col_name.
    """
    tmp_df = pd_frame[numerical_col].groupby(pd_frame.index).aggregate(
        [np.mean, np.std, np.median, stats.skew])
    tmp_df.columns = ['%s_%s'% (numerical_col, i) for i in tmp_df.columns]
    return tmp_df


def main():
    """
    Extract frequency counts from categorical columns and simple distribution
    statistics from numerical ones.
    """
    # Load basic training and testing data, from CSV file.
    sessions = pd.read_csv(SESSSIONS_CSV_FILE)
    sessions.set_index(INDEX_COLUMN, inplace=True)
    sessions.fillna(-1, inplace=True)
    # Extract features from sessions.
    remove_rare_values_inplace(sessions, CATEGORICAL_FEATURES, VALUE_THRESHOLD)
    frequency_counts = extract_frequency_counts(sessions, CATEGORICAL_FEATURES)
    simple_stats = extract_distribution_stats(sessions, SECS_ELAPSED_NUMERICAL)
    # Save new data.
    session_data = pd.concat((frequency_counts, simple_stats), axis=1)
    session_data.fillna(-1, inplace=True)
    session_data.to_csv(OUTPUT_TO_CSV_FILE)


if __name__ == '__main__':
    # Global variables:
    #    INDEX_NAME -- The column that should be set as index.
    #    CATEGORICAL_FEATURES -- The list of categorical features.
    #    SECS_ELAPSED_NUMERICAL -- The name of the numerical column in the data.
    #    VALUE_THRESHOLD -- A parameter to speed-up computation. Categorical
    #        values that appear less than the threshold will be removed.
    #    SESSSIONS_CSV_FILE -- The name of the file that contains the sessions.
    #    OUTPUT_CSV_FILE -- Where to save the extracted features.

    INDEX_COLUMN = 'user_id'
    CATEGORICAL_FEATURES = ['action', 'action_type', 'action_detail',
                            'device_type']
    SECS_ELAPSED_NUMERICAL = 'secs_elapsed'
    VALUE_THRESHOLD = 0.005
    SESSSIONS_CSV_FILE = 'sessions.csv'
    OUTPUT_TO_CSV_FILE = 'ssession_features.csv'

    main() # Run the main method.
