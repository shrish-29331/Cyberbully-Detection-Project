import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_data():
    # Load the Excel file
    #df = pd.read_excel('/kaggle/working/bully_data/Cyberbully_corrected_emotion_sentiment_v2.xlsx')
    excel_dir='bully_data/data/bully_data/Cyberbully_corrected_emotion_sentiment_v2.xlsx'
    img_dir = "bully_data/data/bully_data/"
    
    df = pd.read_excel(excel_dir)
    df = df.drop(columns=['Unnamed: 10', 'Unnamed: 11'])

    df_cleaned = df.dropna()
    df=df_cleaned

    

    # Define a function to check if the image size is zero
    def is_zero_size(img_id, img_dir):
        img_path = os.path.join(img_dir, img_id)
        return os.path.exists(img_path) and os.path.getsize(img_path) == 0

    # Filter out rows with zero-size images
    df['is_zero_size'] = df['Img_Name'].apply(lambda img_id: is_zero_size(img_id, img_dir))
    df_filtered = df[df['is_zero_size'] == False].drop(columns='is_zero_size')

    # Now, df_filtered contains only rows with non-zero-size images
    # print(df_filtered)
    df=df_filtered
    df_cleaned = df[df['Img_Name'] != '2644.jpg']
    df=df_cleaned

    # print("Columns: ", df.columns.tolist())
    # print(f"{df.columns.tolist()[2]}: {df[df.columns.tolist()[2]].unique()} ")
    # print(f"{df.columns.tolist()[3]}: {df[df.columns.tolist()[3]].unique()} ")
    # print(f"{df.columns.tolist()[4]}: {df[df.columns.tolist()[4]].unique()} ")
    # print(f"{df.columns.tolist()[5]}: {df[df.columns.tolist()[5]].unique()} ")
    # print(f"{df.columns.tolist()[6]}: {df[df.columns.tolist()[6]].unique()} ")
    # print(f"{df.columns.tolist()[7]}: {df[df.columns.tolist()[7]].unique()} ")
    # print(f"{df.columns.tolist()[8]}: {df[df.columns.tolist()[8]].unique()} ")
    # print(f"{df.columns.tolist()[9]}: {df[df.columns.tolist()[9]].unique()} ")

    return df
