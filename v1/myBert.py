import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
import config

seed = 1996
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

class MyBert(nn.Module):
    def __init__(self, num_classes=35, alpha=0.5):
        self.alpha = alpha
        super(MyBert, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path).cuda()
        for name, param in self.bert.named_parameters():
            if 'layer.11' in name or 'layer.10' in name or 'layer.9' in name or 'layer.8' in name or 'pooler.dense' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.MLP = nn.Sequential(
            nn.Linear(768, num_classes, bias=True),
        ).cuda()

    def forward(self, x):
        x = x.long()
        x, pooled_x = self.bert(x, output_all_encoded_layers=False)
        x = F.dropout(x, self.alpha, training=self.training)
        x = torch.max(x, dim=1)[0]
        x = self.MLP(x)
        return torch.sigmoid(x)
