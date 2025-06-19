import torch
import yaml
from torch import nn
import torchvision
from torch.nn import functional as F
from models.SwinTransformer.build_model import build_model


class Detector(nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        yaml_path='./models/SwinTransformer/my_config/NASA-Swin-Base.yaml'
        yaml_config=yaml.load(open(yaml_path, 'r'), Loader=yaml.FullLoader)
        self.net=build_model(yaml_config, pretrained=None)
        

    def forward(self,x):
        x=self.net(x)
        return x
    
    def forward_RIDRes(self, x_res, x_rgb):
        B, C, H, W = x_rgb.shape
        chunks_res = torch.chunk(x_res, C, dim=1)
        chunks_rgb = torch.chunk(x_rgb, C, dim=1)
        interleaved_list = [chunk for pair in zip(chunks_rgb, chunks_res) for chunk in pair]
        x=torch.cat(interleaved_list, dim=1)

        x=self.net(x)
        return x
    
    