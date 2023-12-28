from preprocessing import preprocess_data, clear_directory
from PIL import Image
import os

def test_preprocess_data():
    """Tests transformation of ./tests/test_input data folder"""

    input_dir = "./data_processing/tests/test_input"
    output_dir = "./data_processing/tests/test_output"
    width, height = 320,320

    clear_directory(output_dir)
    preprocess_data(input_dir, output_dir, (width, height))
    output_files = os.listdir(output_dir)

    for file in output_files:
        image = Image.open(os.path.join(output_dir, file))

        assert image.size == (width, height)
        assert image.mode == "RGB"


    




    



