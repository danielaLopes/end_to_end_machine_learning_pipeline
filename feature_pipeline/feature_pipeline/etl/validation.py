from great_expectations.core import ExpectationSuite, ExpectationConfiguration

from feature_pipeline.globals import globals


def build_expectation_suite() -> ExpectationSuite:
    """
    Builder used to retrieve an instance of the validation expectation suite.
    """

    expectation_suite_cat_dog_image = ExpectationSuite(
        expectation_suite_name="cat_dog_image_suite"
    )

    column_list = ["label", "datetime_utc", "image_hash"] + [f"px{i}" for i in range(globals.NUM_IMAGE_FEATURES)]

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
            kwargs={"column": "label", "value_set": ["cat", "dog"]},  # Adjust according to your actual labels
        )
    )

    # Image Hash
    expectation_suite_cat_dog_image.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_unique",
            kwargs={"column": "image_hash"},
        )
    )

    # Pixels
    for px_column in column_list[3:]:  # Skipping the first three non-pixel columns
        expectation_suite_cat_dog_image.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": px_column,
                    "min_value": 0,
                    "max_value": 255,  # Assuming 8-bit grayscale images
                },
            )
        )

    return expectation_suite_cat_dog_image