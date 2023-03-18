import torch
import torch.onnx
from model import Classifier, BasicBlock


if __name__ == "__main__":
    mtailor = Classifier(BasicBlock, [2, 2, 2, 2])
    mtailor.load_state_dict(torch.load("./pytorch_model_weights.pth"))
    mtailor.eval()


    dummy_input = torch.randn(1, 3, 224, 224, requires_grad = False)


    torch.onnx.export(mtailor,
        dummy_input,
        "ImageClassifier.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["modelInput"],
        output_names=["modelOutput"],
        dynamic_axes={'modelInput':{0:'batch_size'},
            'modelOutput':{0:'batch_size'}}
    )

    print("Model has been converted to ONNX")