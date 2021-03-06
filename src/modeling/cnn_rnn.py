import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import os

import torchvision
import pretrainedmodels
import timm


class ResNet(nn.Module):
    """
    ResNet, ResNeXt, SENet.

    Args:
        cfg (CfgNode): Encoder configs
    """
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        model_name = cfg.MODEL.ENCODER.NAME
        input_channels = cfg.DATA.INP_CHANNEL
        num_classes = cfg.MODEL.NUM_CLASSES

        # torchhub WSL
        if model_name.endswith("_wsl"):
            backbone = torch.hub.load(
                "facebookresearch/WSL-Images", model_name)
            in_features = backbone.fc.in_features
        # torchvision
        elif model_name == "resnext50_32x4d":
            backbone = torchvision.models.resnext50_32x4d(pretrained=True)
            in_features = backbone.fc.in_features
        # Cadene's pretrainedmodels
        else:
            backbone = pretrainedmodels.__dict__[
                model_name](num_classes=1000, pretrained="imagenet")
            in_features = backbone.last_linear.in_features

        if hasattr(backbone, "layer0"):
            self.layer0 = backbone.layer0
        else:
            layer0_modules = [
                ('conv1', backbone.conv1),
                ('bn1', backbone.bn1),
                ('relu', backbone.relu),
                ('maxpool', backbone.maxpool)
            ]
            self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        del backbone

        if input_channels != 3:
            old_conv_weight = self.layer0.conv1.weight.data
            new_conv = nn.Conv2d(input_channels, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)
            new_conv.weight.data = torch.stack(
                [torch.mean(old_conv_weight, 1)] * input_channels,
                dim=1)
            self.layer0.conv1 = new_conv

        self.fc = nn.Linear(in_features, num_classes)
        nn.init.zeros_(self.fc.bias.data)


    def _features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x); l2 = x
        x = self.layer3(x); l3 = x
        x = self.layer4(x)
        return l2, l3, x

    def forward(self, x):
        _, _, x = self._features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class RecurrentDecoder(nn.Module):
    """
    N-layer "recurrent" cell (e.g. Bi/Unidirectional LSTM/GRU) with
    last Linear module.

    Args:
        cfg (CfgNode): decoder configs.
    """
    def __init__(self, cfg):
        super(RecurrentDecoder, self).__init__()
        name = cfg.MODEL.DECODER.NAME
        in_features = cfg.MODEL.DECODER.IN_FEATURES
        hidden_size = cfg.MODEL.DECODER.HIDDEN_SIZE
        bidirectional = cfg.MODEL.DECODER.BIDIRECT
        recurrent_features = hidden_size * 2 if bidirectional else hidden_size
        num_layers = cfg.MODEL.DECODER.NUM_LAYERS
        dropout = cfg.MODEL.DECODER.DROPOUT
        num_classes = cfg.MODEL.NUM_CLASSES

        if name == "lstm":
            self.recurrent = nn.LSTM(input_size=in_features, hidden_size=hidden_size,
                dropout=dropout, num_layers=num_layers,
                bidirectional=bidirectional, batch_first=True)
        elif name == "gru":
            self.recurrent = nn.GRU(input_size=in_features, hidden_size=hidden_size,
                dropout=dropout, num_layers=num_layers,
                bidirectional=bidirectional, batch_first=True)

        self.fc = nn.Linear(recurrent_features, num_classes)
        nn.init.zeros_(self.fc.bias.data)

    def forward(self, x, seq_len):
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = x.reshape(-1, seq_len, x.size(-1))
        _ , (x,_) = self.recurrent(x)
        # print(x.shape)
        x = self.fc(x)
        x = x.reshape(-1, x.size(-1))
        # print(x.shape)
        return x


class ResNet3D(ResNet):
    """
    3D ResNet.

    Args:
        cfg (CfgNode): encoder and decoder configs.
    """
    def __init__(self, cfg):
        super(ResNet3D, self).__init__(cfg)
        del self.fc
        self.decoder = RecurrentDecoder(cfg)

    def forward(self, x, seq_len):
        _, _, x = self._features(x)
        x = self.decoder(x, seq_len)
        return x



if __name__ == "__main__":
    from yacs.config import CfgNode as CN
    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.ENCODER = CN()
    cfg.MODEL.ENCODER.NAME = "se_resnext50_32x4d"
    cfg.MODEL.DECODER = CN()
    cfg.MODEL.DECODER.NAME = "lstm"
    cfg.MODEL.DECODER.NUM_LAYERS = 1
    cfg.MODEL.DECODER.IN_FEATURES = 2048
    cfg.MODEL.DECODER.HIDDEN_SIZE = 128
    cfg.MODEL.DECODER.BIDIRECT = False
    cfg.MODEL.DECODER.DROPOUT = 0.3
    cfg.MODEL.NUM_CLASSES = 2
    cfg.DATA = CN()
    cfg.DATA.IMG_SIZE = 128
    cfg.DATA.INP_CHANNEL = 3

    model = ResNet3D(cfg)

    x = torch.rand((32, 10, 3, 128, 128))
    bsize, seq_len, c, h, w = x.size()
    x = x.view(bsize * seq_len, c, h, w)

    x = model(x,seq_len)

    print(x.size())