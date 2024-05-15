import pandas as pd
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from great_expectations.dataset import PandasDataset

from feature_pipeline.globals import globals
import feature_pipeline.feature_pipeline.utils as utils

logger = utils.get_logger(__name__)


def build_expectation_suite() -> ExpectationSuite:
    """
    Builder used to retrieve an instance of the validation expectation suite.
    """

    expectation_suite_cat_dog_image = ExpectationSuite(
        expectation_suite_name="cat_dog_image_suite"
    )

    column_list = ["label", "datetime_utc", "image_hash"] + [f"f{i}" for i in range(globals.NUM_IMAGE_FEATURES)]

    # Columns.
    expectation_suite_cat_dog_image.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_columns_to_match_ordered_list",
            kwargs={
                "column_list": column_list
            },
        )
    )
    expectation_suite_cat_dog_image.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_column_count_to_equal", kwargs={"value": len(column_list)}
        )
    )

    # Datetime UTC
    expectation_suite_cat_dog_image.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "datetime_utc"},
        )
    )

    # Label
    expectation_suite_cat_dog_image.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={"column": "label", "value_set": [0, 1]}, #  0 for dog and 1 for cat
        )
    )

    # Image Hash
    expectation_suite_cat_dog_image.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_unique",
            kwargs={"column": "image_hash"},
        )
    )

    # Image features
    # TODO: Right now, these values aren't unbounded
    # for feat_column in column_list[3:]: # Skipping the first three non-pixel columns
    #     expectation_suite_cat_dog_image.add_expectation(
    #         ExpectationConfiguration(
    #             expectation_type="expect_column_values_to_be_between",
    #             kwargs={
    #                 "column": feat_column,
    #                 "min_value": -1,
    #                 "max_value": 1, # Since we used Tanh in the last layer of the Autoencoder
    #             },
    #         )
    #     )

    return expectation_suite_cat_dog_image


def check_validation_failures(data: pd.DataFrame,
                              validation_expectation_suite: ExpectationSuite) -> None:
    """
    Check validation failures before loading features to the feature
    store. This function is used to pre-validate the data in the 
    dataframe. Since feature_group.insert() does the validation automatically 
    and does not specify the errors, we added this step to get a specific list
    of the errors.

    Raises:
    - ValueError: If data does not meet specified conditions in the ExpectationSuite.
    """
    data_ge = PandasDataset(data)
    # Validate the data against our expectation suite
    validation_result = data_ge.validate(expectation_suite=validation_expectation_suite)

    if validation_result.success:
        logger.info("Data validation passed. Proceeding to load data to the feature store.")
    else:
        logger.error("Data validation failed. Review the following details:")
        for result in validation_result.results:
            if not result.success:
                logger.error(f"Failed expectation: {result.expectation_config.expectation_type}")
                logger.error(f"Result details: {result.result}")
                raise ValueError("There are validation failures. Fix them before trying again. Exiting ...")