import pandas as pd
import numpy as np
import re
import pickle
from collections import OrderedDict
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime, timedelta




def apply_threshold(df, thresh):

    ad = df[['subject_id', 'hadm_id', 'admittime', 'dischtime']]
    ad = ad.groupby(['subject_id', 'hadm_id']).agg({
        # 'hadm_id': list,
        'admittime': lambda x: [date.split()[0] for date in x][0], # keep only date
        'dischtime': lambda x: [date.split()[0] for date in x][0]  # keep only date
    }).reset_index()
    # Group by 'subject_id' and count the unique hadm_ids
    subject_counts = ad.groupby('subject_id')['hadm_id'].nunique()

    # Filter subject_ids based on the threshold
    selected_subject_ids = subject_counts[subject_counts >= thresh].index

    # Filter the original DataFrame based on the selected subject_ids
    filtered_df = ad[ad['subject_id'].isin(selected_subject_ids)]

    # print(filtered_df.shape)

    return filtered_df



def map_icd_prefix(icd_code, category_mapping):
    icd_prefix = str(icd_code)[:3]
    if icd_prefix.startswith('E'):
        return 'E'
    elif icd_prefix.startswith('V'):
        return 'V'
    else:
        icd_prefix = int(icd_prefix)
        for key, value in category_mapping.items():
            if key[0] <= icd_prefix <= key[1]:
                return key[0]



def filter_admissions(df, categories):
    df_filtered = df[(df['icd_version'] == 9)].copy()
    # df_filtered.head(10)
    df_filtered['category_code'] = df_filtered['icd_code'].apply(lambda x: map_icd_prefix(x, categories))
    return df_filtered


def compile_admissions(df):
    df_filtered = df.drop_duplicates(subset=['subject_id', 'hadm_id', 'category_code'])
    df_listed = df_filtered.groupby(['subject_id', 'hadm_id']).agg({
        # 'hadm_id': list,
        # 'seq_num': list,
        'category_code': list,
    }).reset_index()

    return df_listed


def merge_codes(df_codes, df_admissions):
    df_expanded = df_codes.drop_duplicates(subset=['subject_id', 'hadm_id']).reset_index(drop=True)
    merged_df = pd.merge(df_expanded, df_admissions, on=['subject_id', 'hadm_id'], how='inner')
    return merged_df


def custom_sort(value):
    # print(type(value))
    if type(value) == str:  # Check if the value is a char
        return float('inf')  # Assign a high value for 'E' and 'V'
    else:
        return int(value)


def get_possible_codes(df):
    unique_values_list = df['category_code'].unique().tolist()
    # print(unique_values_list)

    # print(len(unique_values_list))
    possible_codes = sorted(unique_values_list, key=custom_sort)

    # print(len(possible_codes))

    # print(possible_codes)
    return possible_codes


def process_admissions(df, max_days=50, zero_date='2000-01-01'):
    df = df[['subject_id', 'hadm_id', 'category_code', 'admittime']]
    df_sorted = df.sort_values(by=['subject_id', 'admittime'])
    df_sorted['admittime'] = pd.to_datetime(df['admittime'])
    df_sorted['days_between_admissions'] = df_sorted.groupby('subject_id')['admittime'].diff().dt.days.fillna(0).astype(int)

    mask = df_sorted['days_between_admissions'] > max_days
    flag = mask.groupby(df_sorted['subject_id']).cumsum().astype(bool)
    df_filtered = df_sorted[~flag]

    zero_days_rows = df_filtered['days_between_admissions'] == 0
    df_filtered.loc[zero_days_rows, 'admittime'] = zero_date

    cumulative_days = df_filtered.groupby('subject_id')['days_between_admissions'].cumsum()
    # df_filtered['admittime'] = pd.to_datetime(zero_date) + pd.to_timedelta(cumulative_days, unit='D')
    # Use .loc[] to set values in DataFrame
    df_filtered.loc[:, 'admittime'] = pd.to_datetime(zero_date) + pd.to_timedelta(cumulative_days, unit='D')


    return df_filtered


# Function to preserve order and remove duplicates
def unique_ordered_list(input_list):
    seen = set()
    return list(OrderedDict.fromkeys(x for x in input_list if x not in seen and not seen.add(x)))


def build_history(df):
    # Apply the custom function within each subject group
    df['category_code'] = df.groupby('subject_id')['category_code'].transform(lambda x: x.cumsum())

    # Drop duplicate rows (keeping the first occurrence for each subject)
    df_cumulative = df.drop_duplicates(subset=['subject_id', 'hadm_id']).reset_index(drop=True)

    # Reset the index
    df_cumulative = df_cumulative.reset_index(drop=True)

    # Assuming 'icd_code_list' is the column with lists in your DataFrame
    df_cumulative['category_code'] = df_cumulative['category_code'].apply(unique_ordered_list)

    return df_cumulative


def threshold_admissions(df, threshold=5, total_admissions=5):
    subject_admission_counts = df.groupby('subject_id')['hadm_id'].nunique()
    valid_subject_ids = subject_admission_counts[subject_admission_counts >= threshold].index
    # valid_subject_ids = subject_admission_counts[subject_admission_counts >= 2].index

    df_final = df[df['subject_id'].isin(valid_subject_ids)].reset_index(drop=True)
    # Assuming df_final is your DataFrame
    df_consider = df_final.groupby('subject_id').head(total_admissions)

    # Resetting the index if needed
    df_consider.reset_index(drop=True, inplace=True)

    return df_consider



def make_sequences(df, possible_codes):
    # Print the length of possible codes
    # print(len(possible_codes))

    # Create a MultiLabelBinarizer with specified classes
    mlb = MultiLabelBinarizer(classes=possible_codes)

    # Transform the 'category_code' column into a binary matrix
    diagnosis_matrix = pd.DataFrame(mlb.fit_transform(df['category_code']), columns=mlb.classes_)

    # Concatenate the encoded features with the original DataFrame
    df_encoded = pd.concat([df[['subject_id', 'admittime']], diagnosis_matrix], axis=1)

    # Sort DataFrame by admission date
    df_encoded.sort_values(by=['subject_id', 'admittime'], inplace=True)

    # Handle varying lengths by padding sequences
    max_sequence_length = len(possible_codes) 
    padded_matrix = pad_sequences(df_encoded.iloc[:, 2:].values, maxlen=max_sequence_length, padding='post', truncating='post')

    # Add the padded matrix as additional columns in the DataFrame
    df_encoded.iloc[:, 2:] = padded_matrix

    # Add a column for time difference between consecutive admissions
    df_encoded['time_difference'] = df_encoded.groupby('subject_id')['admittime'].diff().dt.days.fillna(0).astype(int)

    # Print the shape of the resulting DataFrame
    # print(df_encoded.shape)

    # Vector without time difference
    # Create a new DataFrame with 'subject_id', 'admittime', 'time_difference', and 'vector' columns
    df_new = pd.DataFrame({
        'subject_id': df_encoded['subject_id'],
        'admittime': df_encoded['admittime'],
        'time_difference': df_encoded['time_difference'],
        'vector': df_encoded.apply(lambda row: row.values[2:-1], axis=1).tolist()  # Extracting vector columns
    })

    # Print the shape of the new DataFrame
    # print(df_new.shape)

    # Group by 'subject_id' and aggregate 'vector' as a list
    df_matrix = df_new.groupby('subject_id')['vector'].agg(list).reset_index()

    # Rename columns for clarity
    df_matrix = df_matrix.rename(columns={"vector": "admission_matrix"})

    return df_matrix, mlb


ad = pd.read_csv("admissions.csv")
# ad = ad[['subject_id', 'hadm_id', 'admittime', 'dischtime']]
# ad = ad.groupby(['subject_id', 'hadm_id']).agg({
#     # 'hadm_id': list,
#     'admittime': lambda x: [date.split()[0] for date in x][0], # keep only date
#     'dischtime': lambda x: [date.split()[0] for date in x][0]  # keep only date
# }).reset_index()

diagnoses_icd = pd.read_csv("diagnoses_icd.csv")


def do_all(admissions, diagnoses, category_mapping, min_admissions, admissions_consider):
    filtered_ad = apply_threshold(admissions, min_admissions)
    df_filtered = filter_admissions(diagnoses, category_mapping)
    df_listed = compile_admissions(df_filtered)
    merged_df = merge_codes(df_listed, filtered_ad)
    possible_codes = get_possible_codes(df_filtered)
    df_result = process_admissions(merged_df, 50, '2000-01-01')
    df_cumulative = build_history(df_result)
    df_consider = threshold_admissions(df_cumulative, admissions_consider, admissions_consider)
    df_last, binar = make_sequences(df_consider, possible_codes)

    return df_last, binar





# df_last = do_all(ad, diagnoses_icd)
    
# df_last.head(10)