from modeling.preprocessing import preprocess_image, preprocess_data
from PIL import Image
import os
def clear_directory(directory):

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Error: {e}')

def test_preprocess_data():
    input_dir = "./tests/test_input"
    output_dir = "./tests/test_output"
    width, height = 300, 200

    clear_directory(output_dir)
    preprocess_data(input_dir, output_dir, (width, height))
    output_files = os.listdir(output_dir)

    assert len(output_files) == 7

    for file in output_files:
        image = Image.open(os.path.join(output_dir, file))

        assert image.size == (width, height)
        assert image.mode == "RGB"


    




    



