import os
import datetime

ROOT_DIR = "/Users/danielalopes/end_to_end_machine_learning_pipeline/"
DATASET_CSV_FILE = os.path.join(ROOT_DIR, "feature_pipeline/dataset/cat_or_dog.csv")

END_DATE = datetime.datetime(2024, 6, 11, 14, 0, 0 )
START_DATE = datetime.datetime(2024, 3, 20, 1, 0, 0)

NUM_IMAGE_FEATURES = 32