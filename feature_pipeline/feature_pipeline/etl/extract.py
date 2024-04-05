import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd

from feature_pipeline.globals import globals
import feature_pipeline.feature_pipeline.utils as utils


logger = utils.get_logger(__name__)


def from_file(
    export_end_reference_datetime: Optional[datetime.datetime] = None,
    days_delay: int = 15,
    days_export: int = 30,
    datetime_format: str = "%Y-%m-%d %H:%M"
) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Extract data from the cat_or_dog.csv file.
    
    Args:
        export_end_reference_datetime: The end reference datetime of the export window. If None, the current time is used.
            Because the data is always delayed with "days_delay" days, this date is used only as a reference point.
            The real extracted window will be computed as [export_end_reference_datetime - days_delay - days_export, export_end_reference_datetime - days_delay].
        days_delay: Data has a delay of N days. Thus, we have to shift our window with N days.
        days_export: The number of days to export.
        datetime_format: The datetime format of the fields from the file.


    Returns:
          A tuple of a Pandas DataFrame containing the exported data and a dictionary of metadata.
    """

    export_start, export_end = _compute_extraction_window(export_end_reference_datetime=export_end_reference_datetime, 
                                                          days_delay=days_delay, 
                                                          days_export=days_export)
    records = _extract_records_from_file(export_start=export_start, 
                                         export_end=export_end, 
                                         datetime_format=datetime_format)
    
    metadata = {
        "days_delay": days_delay,
        "days_export": days_export,
        "export_datetime_utc_start": export_start.strftime(datetime_format),
        "export_datetime_utc_end": export_end.strftime(datetime_format),
        "datetime_format": datetime_format,
        "num_unique_samples_per_time_series": len(records["timestamps"].unique()),
    }

    return records, metadata


def _extract_records_from_file(export_start: datetime.datetime, 
                               export_end: datetime.datetime, 
                               datetime_format: str) -> Optional[pd.DataFrame]:
    """Extract records from the file backup based on the given export window."""

    try:
        data = pd.read_csv(globals.DATASET_CSV_FILE)
    except EmptyDataError:        
        raise ValueError(f"Downloaded file at {globals.DATASET_CSV_FILE} is empty. Could not load it into a DataFrame.")

    # Filter rows that are within the expected time window
    records = data[(data["timestamps"] >= export_start.strftime(datetime_format)) & (data["timestamps"] < export_end.strftime(datetime_format))]
    logger.info(f"""Label distribution of data collected between {export_start} and {export_end}:\n \
                {records['label'].value_counts()}""")
    return records


def _compute_extraction_window(export_end_reference_datetime: datetime.datetime, 
                               days_delay: int, 
                               days_export: int) -> Tuple[datetime.datetime, 
                                                          datetime.datetime]:
    """
    Compute the extraction window relative to 'export_end_reference_datetime' 
    and take into consideration the maximum and minimum data points available in the dataset.
    """

    if export_end_reference_datetime is None:
        export_end_reference_datetime = globals.END_DATE + datetime.timedelta(days=days_delay)
        export_end_reference_datetime = export_end_reference_datetime.replace(
            minute=0, second=0, microsecond=0
        )
    else:
        export_end_reference_datetime = export_end_reference_datetime.replace(
            minute=0, second=0, microsecond=0
        )

    export_end = export_end_reference_datetime - datetime.timedelta(days=days_delay)
    export_start = export_end_reference_datetime - datetime.timedelta(
        days=days_delay + days_export
    )

    min_export_start = globals.START_DATE
    if export_start < min_export_start:
        export_start = min_export_start
        export_end = export_start + datetime.timedelta(days=days_export)

    return export_start, export_end