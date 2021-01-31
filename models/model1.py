import torch
import torch.nn as nn
import torch.nn.functional as F
from  models.blocks import  ConvBlock, LinearAttentionBlock, ProjectorBlock
from models.initialize import *
from torch.hub import load_state_dict_from_url
from collections import OrderedDict



def load_model(code_length):
    """
    Load CNN model.

    Args
        code_length (int): Hashing code length.

    Returns
        model (torch.nn.Module): CNN model.
    """
    #model = (AttnVGG_before(code_length, attention = False))
    model = nn.DataParallel(AttnVGG_before(code_length, attention = False))
    state_dict = torch.load("data/net_noatt.pth")  #   net_att_bef.pth OR net_att_bef.pth
    model.load_state_dict(state_dict,  strict=False)
    
    #model = AlexNet(code_length)
    #state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    #model.load_state_dict(state_dict, strict=False)
    
    num_ftrs = model.module.classify.in_features
    model.module.classify = nn.Sequential(
            nn.Linear(num_ftrs, code_length),
            nn.Tanh()
           )
            
    return  model

'''
attention before max-pooling
'''

class AttnVGG_before(nn.Module):
    def __init__(self, code_length, attention=False, init='xavierUniform'):
        super(AttnVGG_before, self).__init__()
        self.attention = attention
        print(attention)
        self.size = 32
        # conv blocks
        self.conv_block1 = ConvBlock(3, 64, 2)
        self.conv_block2 = ConvBlock(64, 128, 2)
        self.conv_block3 = ConvBlock(128, 256, 3)
        self.conv_block4 = ConvBlock(256, 512, 3)
        self.conv_block5 = ConvBlock(512, 512, 3)
        self.conv_block6 = ConvBlock(512, 512, 2, pool=True)
        self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(self.size/32), padding=0, bias=True)
        # Projectors & Compatibility functions
        if self.attention:
            self.projector = ProjectorBlock(256, 512)
            self.attn1 = LinearAttentionBlock(in_features=512, normalize_attn=True)
            self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=True)
            self.attn3 = LinearAttentionBlock(in_features=512, normalize_attn=True)
            
            
        # final classification layer
        if self.attention:
            self.classify = nn.Linear(in_features=512*3, out_features=100, bias=True)

        else:
            self.classify = nn.Linear(in_features=512, out_features=100, bias=True)
            
        # initialize
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self)
        else:
            raise NotImplementedError("Invalid type of initialization!")
    def forward(self, x):
        # feed forward
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        l1 = self.conv_block3(x) # /1
        x = F.max_pool2d(l1, kernel_size=2, stride=2, padding=0) # /2
        l2 = self.conv_block4(x) # /2
        x = F.max_pool2d(l2, kernel_size=2, stride=2, padding=0) # /4
        l3 = self.conv_block5(x) # /4
        x = F.max_pool2d(l3, kernel_size=2, stride=2, padding=0) # /8
        x = self.conv_block6(x) # /32
        g = self.dense(x) # batch_sizex512x1x1
        # pay attention
        if self.attention:       
            c1, g1 = self.attn1(self.projector(l1), g)
            c2, g2 = self.attn2(l2, g)
            c3, g3 = self.attn3(l3, g)
            g = torch.cat((g1,g2,g3), dim=1) # batch_sizexC
            # classification layer
            x = self.classify(g) # batch_sizexnum_classes
        else:
            c1, c2, c3 = None, None, None
            x = self.classify(torch.squeeze(g))
        return x                     #[x, c1, c2, c3] change the model output for simplicity


class AlexNet(nn.Module):

    def __init__(self, code_length):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096 ,1000),
        )

        self.classifier = self.classifier[:-1]
        self.hash_layer = nn.Sequential(
            nn.Linear(4096, code_length),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.hash_layer(x)
        return x