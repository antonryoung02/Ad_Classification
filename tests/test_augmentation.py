from data_processing.preprocessing import clear_directory
from modeling.Augment import AbstractTransformation, AugmentedImageFolder
from torchvision.transforms import v2
from torchvision.transforms.functional import to_pil_image
from modeling.Augment import AugmentationFactory
from modeling.utils import load_config

def simulate_data_augmentation_method(augmentation:AbstractTransformation, invert_normalization:bool=False):
    """Helps test quality of the transformations in test_output directory"""
    input_dir = "./tests/test_input"
    output_dir = "./tests/test_output"
    clear_directory(output_dir)

    train_transform  = v2.Compose([v2.ToTensor()])
    invert_train_transform = v2.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], #Inverts imagenet normalization constants
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    train_data_folder = AugmentedImageFolder(root=input_dir, transform=train_transform, augmentation=augmentation)
    i = 0
    
    for augmented_tensor in train_data_folder:
        if invert_normalization:
            augmented_tensor = invert_train_transform(augmented_tensor)
        augmented_image = to_pil_image(augmented_tensor[0])
        augmented_image.save(f"{output_dir}/image{i}.png")
        i += 1

# def view_image_histograms():
#     input_dir = "./tests/test_input"
#     train_transform  = v2.ToTensor()
#     invert_train_transform = v2.Normalize(
#         mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], #Inverts imagenet normalization constants
#         std=[1/0.229, 1/0.224, 1/0.225]
#     )
#     train_data_folder = AugmentedImageFolder(root=input_dir, transform=train_transform, augmentation=None)

#     for i, augmented_tensor in enumerate(train_data_folder):
#         # print(augmented_tensor[0][0].shape)
#         # break
#         plt.hist(augmented_tensor[0][2].reshape(-1), bins=100)
#         plt.title(f"B;ue channel {'Hockey' if augmented_tensor[1] == 0 else 'Ad'}")
#         plt.subplot(2,5,i+1)
        
#     plt.show()
    
# def view_image_intensity_profile():
#     input_dir = "./tests/test_input"
#     train_transform  = v2.ToTensor()
#     invert_train_transform = v2.Normalize(
#         mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], #Inverts imagenet normalization constants
#         std=[1/0.229, 1/0.224, 1/0.225]
#     )
#     train_data_folder = AugmentedImageFolder(root=input_dir, transform=train_transform, augmentation=None)

#     for i, augmented_tensor in enumerate(train_data_folder):
#         # 27th column intensity profile
#         if i < 9:
#             plt.plot(augmented_tensor[0][0][26].reshape(-1))
#             plt.title(f"Red channel {'Hockey' if augmented_tensor[1] == 0 else 'Ad'}")
#             plt.subplot(3,3,i+1)
#     plt.show()
    
    
def main():
    config = load_config('modeling/configs/2024-12-02_comic-sweep-2.yaml')
    dt = AugmentationFactory()(config)
    simulate_data_augmentation_method(dt, invert_normalization=True)
    #view_image_histograms()
    # view_image_intensity_profile()

if __name__ == "__main__":
    main()




    



