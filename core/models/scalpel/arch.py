import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class LesionScalpel(nn.Module):
    def __init__(self):
        super(LesionScalpel, self).__init__()
        
        self.model = smp.Unet(
            encoder_name="resnet34",        # encoder
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
            activation="sigmoid"            # activation function to apply after final convolution;
        )

    def forward(self, x):
        return self.model(x)
