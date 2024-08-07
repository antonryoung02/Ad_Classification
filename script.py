import torch
import coremltools as ct
from modeling.utils import SqueezeNetWithSkipConnections
from modeling.CNN import CNN

# https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html
def convert_pt_to_coreml(model_path:str, new_name:str):
    CONFIG = {
        "base_e": 128,
        "batch_size":256,
        "dropout":0.06482,
        "incr_e":96,
        "initializer":"squeezenet",
        "lr":0.01786,
        "lr_gamma":0.3295,
        "num_epochs":5,
        "pct_3x3":0.5,
        "sr":0.25,
        "weight_decay":0.001374
    }
    model = SqueezeNetWithSkipConnections(config=CONFIG, input_shape=(3,224,224))
    model.load_state_dict(torch.load(model_path))
   # model = torch.load(model_path)
    model.eval()
    example_input = torch.rand(1, 3, 224, 224) 
    traced_model = torch.jit.trace(model, example_input)
    out = traced_model(example_input)

    scale = 1.0 / (255.0 * 0.226)
    red_bias = -0.485 / 0.226
    green_bias = -0.456 / 0.226
    blue_bias = -0.406 / 0.226
    input = ct.ImageType(name='input', bias=[red_bias, green_bias, blue_bias], scale=scale, shape=(1, 3, 224, 224), color_layout='RGB')

    
    core_model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[input]
    )
    core_model.save(f"{new_name}.mlpackage")

convert_pt_to_coreml("./model.pt", "core_model")