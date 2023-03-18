import torch
from PIL import Image
from model import Classifier, BasicBlock, TestOnnx
import numpy as np


def test_onnx_func(image):
    mtailor = Classifier(BasicBlock, [2, 2, 2, 2])
    mtailor.load_state_dict(torch.load("./pytorch_model_weights.pth"))
    mtailor.eval()


    img = Image.open(f"./{image}")
    
    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()


    inp = mtailor.preprocess_numpy(img).unsqueeze(0)

    test_onnx = TestOnnx()
    tensor_image = test_onnx.output_prediction(inp)
    tensor_image = Image.fromarray(np.uint8((tensor_image[0] * 255.0).clip(0, 255)[0]))

    final_img = Image.merge(
        "YCbCr", [
            tensor_image,
            img_cb.resize(tensor_image.size, Image.BILINEAR),
            img_cr.resize(tensor_image.size, Image.BILINEAR)
        ]
    ).convert("RGB")


    final_img.save("./temp.jpeg")
