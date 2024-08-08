import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import dotenv
dotenv_path = os.path.join(root_dir, '.env')
dotenv.load_dotenv(dotenv_path)

from data_processing.preprocessing import clear_directory, preprocess_data
from data_processing.openverse import Openverse, OpenverseQuery
from data_processing.metadata_dataframe import MetadataDataframe
# OUTPUT_POS = "./modeling/data/data_pos"
# OUTPUT_NEG = "./modeling/data/data_neg"
# IMAGE_DIMENSION = (224,224)
# clear_directory(OUTPUT_POS) #For processing reusability
# clear_directory(OUTPUT_NEG)

# #Handle data sources differently
# preprocess_data("./raw_data/new_data_pos", OUTPUT_POS, dimensions= IMAGE_DIMENSION)
# preprocess_data("./raw_data/new_data_neg", OUTPUT_NEG, dimensions=IMAGE_DIMENSION)
# preprocess_data("./raw_data/data_pos", OUTPUT_POS, dimensions=IMAGE_DIMENSION)
# preprocess_data("./raw_data/data_neg", OUTPUT_NEG, dimensions= IMAGE_DIMENSION)
# preprocess_data("./raw_data/collected_data_pos", OUTPUT_POS, dimensions=IMAGE_DIMENSION)
# preprocess_data("./raw_data/collected_data_neg", OUTPUT_NEG, dimensions= IMAGE_DIMENSION)
SEARCH_QUERY_PATH = os.getenv('SEARCH_QUERY_PATH')
DATASET_DIRECTORY = os.getenv('DATASET_DIRECTORY')
METADATA_PATH = os.getenv('METADATA_PATH')
OPENVERSE_CREDENTIALS_PATH = os.getenv("OPENVERSE_CREDENTIALS_PATH")

meta_df = MetadataDataframe(METADATA_PATH)
ov = Openverse(OPENVERSE_CREDENTIALS_PATH, metadata_dataframe=meta_df)
#ov.register()
#ov.verify_email()
token = ov.get_access_token()
query_params = {"page_size":50, "page":3}

search_queries = []
with open(SEARCH_QUERY_PATH, 'r') as file:
    search_queries = [line.strip() for line in file if line.strip()]

for query_str in search_queries:
    print(f"Query for {query_str}")
    query = OpenverseQuery(query_str=query_str, query_params=query_params)
    images = ov.get_images(query, token)
    if images is not None:
        ov.save_images_to_directory(images, DATASET_DIRECTORY)