import datetime
from typing import Optional
import fire
import pandas as pd
import traceback

from etl import cleaning, load, extract, validation
import utils

logger = utils.get_logger(__name__)


def run(
    export_end_reference_datetime: Optional[datetime.datetime] = None,
    days_delay: int = 10,
    days_export: int = 60,
    feature_group_version: int = 2,
) -> dict:
    """
    Extract data from the csv file, transform it, and load it to the feature store.

    Args:
        export_end_reference_datetime: The end reference datetime of the export window. If None, the current time is used.
            Because the data is always delayed with "days_delay" days, this date is used only as a reference point.
            The real extracted window will be computed as [export_end_reference_datetime - days_delay - days_export, export_end_reference_datetime - days_delay].
        days_delay: Data has a delay of N days. Thus, we have to shift our window with N days.
        days_export: The number of days to export.
        feature_group_version: The version of the feature store feature group to save the data to.

    Returns:
          A dictionary containing metadata of the pipeline.
    """

    logger.info(f"Extracting data from cat_or_dog.csv file.")
    data, metadata = extract.from_file(
        export_end_reference_datetime, days_delay, days_export
    )
    if metadata["num_unique_samples_per_time_series"] < days_export * 24:
        raise RuntimeError(
            f"Could not extract the expected number of samples from the cat_or_dog.csv file: \
                {metadata['num_unique_samples_per_time_series']} < {days_export * 24}."
        )
    logger.info("Successfully extracted data from API.")

    logger.info(f"Transforming data.")
    data = transform(data)
    logger.info("Successfully transformed data.")

    logger.info("Building validation expectation suite.")
    validation_expectation_suite = validation.build_expectation_suite()
    logger.info("Successfully built validation expectation suite.")

    try:
        logger.info(f"Check validation failures.")
        validation.check_validation_failures(data, 
                                            validation_expectation_suite)
        logger.info(f"Successfully checked validation failures.")
    except ValueError as e:
        logger.error(traceback.format_exc())
        exit(1)

    logger.info(f"Validating data and loading it to the feature store.")
    load.to_feature_store(
        data,
        validation_expectation_suite=validation_expectation_suite,
        feature_group_version=feature_group_version,
    )
    metadata["feature_group_version"] = feature_group_version
    logger.info("Successfully validated data and loaded it to the feature store.")

    logger.info(f"Wrapping up the pipeline.")
    utils.save_json(metadata, file_name="feature_pipeline_metadata.json")
    logger.info("Done!")

    return metadata


def transform(data: pd.DataFrame):
    """
    Wrapper containing all the transformations from the ETL pipeline.
    """

    data = cleaning.rename_columns(data)
    data = cleaning.cast_columns(data)
    data = cleaning.generate_image_hash(data)
    data = cleaning.remove_duplicate_images(data)
    logger.info(f"Transformed data: {data}")

    return data


if __name__ == "__main__":
    fire.Fire(run)