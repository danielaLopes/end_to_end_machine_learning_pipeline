import pandas as pd
import hashlib

import utils
from feature_pipeline.globals import globals
logger = utils.get_logger(__name__)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to match our schema.
    """

    data = df.copy()

    # Rename columns
    data.rename(
        columns={
            "timestamps": "datetime_utc",
        },
        inplace=True,
    )
    # Feature names need to start with a letter
    columns_to_rename = [str(i) for i in range(globals.NUM_IMAGE_FEATURES) if str(i) in data.columns]
    rename_dict = {col: f'f{col}' for col in columns_to_rename}

    data.rename(columns=rename_dict, 
                inplace=True)

    return data


def cast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast columns to the correct data type.
    """

    data = df.copy()

    data["datetime_utc"] = pd.to_datetime(data["datetime_utc"])

    return data


def _hash_image_pixels(row):
    pixel_data_str = ''.join(map(str, row.values))
    # Convert the string representation to bytes
    pixel_data_bytes = pixel_data_str.encode()
    hash_value = hashlib.sha256(pixel_data_bytes).hexdigest()
    
    return hash_value


def generate_image_hash(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates an hash for every image to be used as their unique 
    identifiers.
    """

    data = df.copy()
    
    pixel_columns = [f'f{i}' for i in range(globals.NUM_IMAGE_FEATURES) if f'f{i}' in data.columns]
    data['image_hash'] = data[pixel_columns].apply(_hash_image_pixels, axis=1)

    # Reorder columns so that feature columns come last
    data = data[['label', 'datetime_utc', 'image_hash'] + pixel_columns]

    return data


def remove_duplicate_images(df: pd.DataFrame, column_name='image_hash') -> pd.DataFrame:
    """
    Removes rows with duplicate values in the specified column.

    Parameters:
    - df: pd.DataFrame - The DataFrame to process.
    - column_name: str - The name of the column to check for duplicate values.
      Default is 'image_hash'.

    Returns:
    - A new DataFrame with duplicates removed.
    """
    # Remove duplicates, keeping the first occurrence
    unique_df = df.drop_duplicates(subset=[column_name], keep='first')
    return unique_df
