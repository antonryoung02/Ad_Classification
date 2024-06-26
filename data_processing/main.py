import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from data_processing.preprocessing import clear_directory, preprocess_data

OUTPUT_POS = "./modeling/data/data_pos"
OUTPUT_NEG = "./modeling/data/data_neg"
IMAGE_DIMENSION = (224,224)
clear_directory(OUTPUT_POS) #For processing reusability
clear_directory(OUTPUT_NEG)

#Handle data sources differently
preprocess_data("./raw_data/new_data_pos", OUTPUT_POS, dimensions= IMAGE_DIMENSION)
preprocess_data("./raw_data/new_data_neg", OUTPUT_NEG, dimensions=IMAGE_DIMENSION)
preprocess_data("./raw_data/data_pos", OUTPUT_POS, dimensions=IMAGE_DIMENSION)
preprocess_data("./raw_data/data_neg", OUTPUT_NEG, dimensions= IMAGE_DIMENSION)
preprocess_data("./raw_data/collected_data_pos", OUTPUT_POS, dimensions=IMAGE_DIMENSION)
preprocess_data("./raw_data/collected_data_neg", OUTPUT_NEG, dimensions= IMAGE_DIMENSION)


