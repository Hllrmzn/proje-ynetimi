import os
import torch
import albumentations

import numpy as np
import pandas as pd

import torch.nn as nn
from sklearn import metrics
from sklearn import model_selection
from torch.nn import functional as F

from wtfml.utils import EarlyStopping
from wtfml.engine import Engine
from wtfml.data_loaders.image import segmentation

import pretrainedmodels


class SEResnext50_32x4d(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super(SEResnext50_32x4d, self).__init__()
        
        self.base_model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=None)
        if pretrained is not None:
            self.base_model.load_state_dict(
                torch.load(
                    "../input/pretrained-model-weights-pytorch/se_resnext50_32x4d-a260b3a4.pth"/// resnext50 proje kaggle üzerinden çalıştırıldığı için path bu şekil
                )
            )

        self.l0 = nn.Linear(2048, 1)
    def forward(self, image, targets):
        batch_size, _, _, _ = image.shape
        
        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        
        out = self.l0(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(x))

        return out, loss
df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
df["kfold"] = -1    
df = df.sample(frac=1).reset_index(drop=True)
y = df.target.values
kf = model_selection.StratifiedKFold(n_splits=5)
