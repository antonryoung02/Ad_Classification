from preprocessing import preprocess_data, clear_directory
from PIL import Image
import os

def test_preprocess_data():
    """Tests transformation of ./tests/test_input data folder"""
    input_dir = "./data_processing/tests/test_input"
    output_dir = "./data_processing/tests/test_output"
    width, height = 320,320
    input_files = os.listdir(input_dir)
    input_png_files = 0
    for file in input_files:
        if file.lower().endswith('.png'):
            input_png_files += 1

    clear_directory(output_dir)
    preprocess_data(input_dir=input_dir, output_dir=output_dir, augment_input=False, dimensions=(width, height))
    output_files = os.listdir(output_dir)
    assert len(output_files) == input_png_files

    clear_directory(output_dir)
    preprocess_data(input_dir=input_dir, output_dir=output_dir, augment_input=True, dimensions=(width, height))
    output_files = os.listdir(output_dir)

    assert len(output_files) == 4 * input_png_files
    for file in output_files:
        image = Image.open(os.path.join(output_dir, file))
        assert image.size == (width, height)
        assert image.mode == "RGB"
        assert file.lower().endswith('.png')


    




    



