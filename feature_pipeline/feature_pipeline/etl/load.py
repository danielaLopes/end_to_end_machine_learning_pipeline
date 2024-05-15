import hopsworks
import pandas as pd
from great_expectations.core import ExpectationSuite
from hsfs.feature_group import FeatureGroup

from feature_pipeline.feature_pipeline.settings import SETTINGS
from feature_pipeline.globals import globals
import feature_pipeline.feature_pipeline.utils as utils

logger = utils.get_logger(__name__)


def to_feature_store(
    data: pd.DataFrame,
    validation_expectation_suite: ExpectationSuite,
    feature_group_version: int,
) -> FeatureGroup:
    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a
    feature store in the feature store.
    """

    # Connect to feature store.
    project = hopsworks.login(
        api_key_value=SETTINGS["FS_API_KEY"], project=SETTINGS["FS_PROJECT_NAME"]
    )
    feature_store = project.get_feature_store()

    # Create feature group.
    feature_group = feature_store.get_or_create_feature_group(
        name=SETTINGS["FS_GROUP_NAME"],
        version=feature_group_version,
        description="Cat or dog image data with timestamps. Data is uploaded with a \
                    10 days delay, and it contains the data of 60 days.",
        primary_key=["image_hash"],
        event_time="datetime_utc",
        online_enabled=False,
        expectation_suite=validation_expectation_suite,
    )
    logger.info(data)
    # Upload data.
    feature_group.insert(
        features=data,
        overwrite=False,
        write_options={
            "wait_for_job": True,
        },
    )

    # Add feature descriptions.
    feature_descriptions = [
        {
            "name": "datetime_utc",
            "description": """
                            The timestamp when the image was captured or processed, in UTC.
                            """,
            "validation_rules": "Should match the datetime format 'YYYY-MM-DD HH:MM:SS'",
        },
        {
            "name": "label",
            "description": """
                            The category or class the image belongs to.
                            """,
            "validation_rules": "=0 for dogs and =1 for cats (integer).",
        },
        {
            "name": "image_hash",
            "description": """
                            A unique hash of the image, generated based on its pixel content.
                            """,
            "validation_rules": "Should be a hexadecimal string representing the SHA-256 hash of the image's pixels.",
        }
    ]

    # Assuming your dataset includes images represented as flattened pixel arrays,
    # you might also want to describe a generalized pixel feature:
    feature_template = {
        "name": "f{index}",
        "description": "Feature {index} as a floating-point value.",
        "validation_rules": "Any float value",
    }

    # Dynamically generate pixel feature descriptions for a 100x100 image
    for i in range(globals.NUM_IMAGE_FEATURES):  # Assuming images are 100x100 pixels, flattened
        feature = {
            "name": feature_template["name"].format(index=i),
            "description": feature_template["description"].format(index=i),
            "validation_rules": feature_template["validation_rules"],
        }
        feature_descriptions.append(feature)

    for description in feature_descriptions:
        feature_group.update_feature_description(
            description["name"], description["description"]
        )

    # Update statistics.
    feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    feature_group.update_statistics_config()
    feature_group.compute_statistics()

    return feature_group