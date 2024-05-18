import os
from PIL import Image
from  modeling.models import DeepCNN
# input_folder = "./raw_data/mac_data_neg"


# output_folder = "./raw_data/mac_data_neg"
# new_size = (400, 400)

# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# for filename in os.listdir(input_folder):
#     if filename.endswith((".jpg", ".jpeg", ".png")):
#         try:
#             image_path = os.path.join(input_folder, filename)
#             img = Image.open(image_path)
#             img = img.resize(new_size)
#             output_path = os.path.join(output_folder, filename)
#             img.save(output_path)
#         except:
#             print(f"Image {filename} is truncated")


model = DeepCNN()
print(model.find_fc_layer_input_shape())
