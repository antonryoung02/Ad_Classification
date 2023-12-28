from preprocessing import clear_directory, preprocess_data

OUTPUT_POS = "./modeling/data/data_pos"
OUTPUT_NEG = "./modeling/data/data_neg"

clear_directory(OUTPUT_POS) #For processing reusability
clear_directory(OUTPUT_NEG)

preprocess_data("./raw_data/mac_data_pos", OUTPUT_POS)
preprocess_data("./raw_data/mac_data_neg", OUTPUT_NEG)
preprocess_data("./raw_data/pi_data_pos", OUTPUT_POS)
preprocess_data("./raw_data/pi_data_neg", OUTPUT_NEG)
