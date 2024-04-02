import os
import numpy as np
import pandas as pd
from PIL import Image
import opendatasets as od
from tqdm import tqdm
import locale

import sys
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.append(base_dir)
from globals import globals


def save_df_in_chunks(df, csv_file_path, chunk_size=100):
    """
    Save a DataFrame to a CSV file in chunks to reduce memory usage.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        csv_file_path (str): The path to the CSV file to save the DataFrame to.
        chunk_size (int): The number of rows per chunk.
    """

    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)
    
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as f:
        for i in range(0, len(df), chunk_size):
            df_chunk = df[i:i+chunk_size]
            # Write the header only on the first chunk
            header = (i == 0)
            df_chunk.to_csv(f, header=header, index=False)


def images_to_csv(image_folder, csv_file_path):
    """
    Read images from a folder, flatten them, and save the data into a CSV file.

    Args:
        image_folder (str): Path to the folder containing images.
        csv_file_path (str): Path to save the CSV file containing flattened image data.
    """

    if not os.path.exists(image_folder):
        od.download("https://www.kaggle.com/datasets/tongpython/cat-and-dog/data")

    datasets = ['training_set/training_set', 'test_set/test_set']
    image_data = []
    image_labels = []
    possible_labels = ['cats', 'dogs']

    for dataset in datasets:
        for i, label in enumerate(possible_labels):
            folder_path = os.path.join(image_folder, dataset, label)
            files = os.listdir(folder_path)
            for file in tqdm(files, total=len(files)):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, file)
                    image = Image.open(image_path).convert('L') # Convert to grayscale
                    image = image.resize((20, 20))
                    image_array = np.array(image).flatten()
                    image_data.append(image_array)
                    image_labels.append(i)


    df = pd.DataFrame(image_data)
    df['label'] = image_labels
    # OutOfMemory error when saving the entire DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)
    #save_df_in_chunks(df, csv_file_path)


if __name__ == '__main__':
    images_to_csv('cat-and-dog', globals.DATASET_CSV_FILE)
