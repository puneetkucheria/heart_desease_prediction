import sqlite3
import pandas as pd


def get_heart_data():
    # import data from sqlite3 database
    con = sqlite3.connect('./../../../Database.db')
    df = pd.read_sql_query("SELECT * FROM Heart_disease", con)
    return df

def convert_num_col(df, num_col):
    for col in num_col:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def map_cat_col(df):
    map_sex = {'Female': 0, 'Male': 1}
    map_yn = {'No': 0, 'Yes': 1}
    df['HeartDisease'] = df['HeartDisease'].map(map_yn)
    df['Smoking'] = df['Smoking'].map(map_yn)
    df['AlcoholDrinking'] = df['AlcoholDrinking'].map(map_yn)
    df['Stroke'] = df['Stroke'].map(map_yn)
    df['DiffWalking'] = df['DiffWalking'].map(map_yn)
    df['Sex'] = df['Sex'].map(map_sex)
    df['PhysicalActivity'] = df['PhysicalActivity'].map(map_yn)
    df['Asthma'] = df['Asthma'].map(map_yn)
    df['KidneyDisease'] = df['KidneyDisease'].map(map_yn)
    df['SkinCancer'] = df['SkinCancer'].map(map_yn)
    return df

def replace_missing_data(df):
    df['Smoking'] = df['Smoking'].fillna(0)
    df['DiffWalking'] = df['DiffWalking'].fillna(0)
    df['Sex'] = df['Sex'].fillna(1)
    df['Asthma'] = df['Asthma'].fillna(0)
    df['SkinCancer'] = df['SkinCancer'].fillna(0)
    df['Diabetic'] = df['Diabetic'].fillna(0)
    df['GenHealth'] = df['GenHealth'].replace('','Very good')
    df['BMI'] = df['BMI'].fillna(df['BMI'].median())
    df['PhysicalHealth'] = df['PhysicalHealth'].fillna(df['PhysicalHealth'].median())
    return df

def get_dummies(df):
    col_dumm = ['AgeCategory', 'Race', 'Diabetic', 'GenHealth' ]
    df = pd.get_dummies(df, columns =col_dumm, drop_first= False)
    return df