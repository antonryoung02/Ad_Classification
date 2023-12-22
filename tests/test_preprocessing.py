from modeling.preprocessing import preprocess_image, preprocess_data
from PIL import Image
import os

def test_preprocess_data():
    input_dir = "./tests/test_input"
    output_dir = "./tests/test_output"

    width, height = 300, 200
    preprocess_data(input_dir, output_dir, (width, height))

    output_files = os.listdir(output_dir)
    print(type(output_files))

    assert len(output_files) == 5

    for file in output_files:
        image = Image.open(os.path.join(output_dir, file))

        assert image.size == (width, height)
        assert image.mode == "RGB"


    




    



