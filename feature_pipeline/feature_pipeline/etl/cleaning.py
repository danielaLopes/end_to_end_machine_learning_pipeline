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
    rename_dict = {col: f'px{col}' for col in columns_to_rename}

    data.rename(columns=rename_dict, 
                inplace=True)
    
    logger.info(f"New columns: {data.columns}")

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
    
    pixel_columns = [f'px{i}' for i in range(globals.NUM_IMAGE_FEATURES) if f'px{i}' in data.columns]
    data['image_hash'] = data[pixel_columns].apply(_hash_image_pixels, axis=1)

    return data