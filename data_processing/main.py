import os
from data_processing.preprocessing import clear_directory, preprocess_data
from data_processing.openverse import Openverse, OpenverseQuery
from data_processing.metadata_dataframe import MetadataDataframe
from data_processing.synthetic_data import SyntheticData, OpenAIProvider

SEARCH_QUERY_PATH = os.getenv('SEARCH_QUERY_PATH')
DATASET_DIRECTORY = os.getenv('DATASET_DIRECTORY')
METADATA_PATH = os.getenv('METADATA_PATH')
OPENVERSE_CREDENTIALS_PATH = os.getenv("OPENVERSE_CREDENTIALS_PATH")

meta_df = MetadataDataframe(METADATA_PATH)
ov = Openverse(OPENVERSE_CREDENTIALS_PATH, metadata_dataframe=meta_df)
#ov.register()
#ov.verify_email()
token = ov.get_access_token()
query_params = {"page_size":50, "page":2}

search_queries = []
with open(SEARCH_QUERY_PATH, 'r') as file:
    search_queries = [line.strip() for line in file if line.strip()]

for query_str in search_queries:
    print(f"Query for {query_str}")
    query = OpenverseQuery(query_str=query_str, query_params=query_params)
    images = ov.get_images(query, token)
    if images is not None:
        ov.save_images_to_directory(images, DATASET_DIRECTORY)


# openai_provider = OpenAIProvider("dall-e-3")
# with open(SEARCH_QUERY_PATH, 'r') as file:
#     queries = [line.strip() for line in file if line.strip()]
# synthetic_data = SyntheticData(queries=queries, metadata_dataframe=meta_df, save_dir = DATASET_DIRECTORY)

# images = synthetic_data.fetch(openai_provider)
# for image in images:
#     synthetic_data.download_image(image, DATASET_DIRECTORY)
    

    
    