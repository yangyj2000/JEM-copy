# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import norms
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, norm=None, leak=.2):
        super(wide_basic, self).__init__()
        self.lrelu = nn.LeakyReLU(leak)
        self.bn1 = get_norm(in_planes, norm)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = Identity() if dropout_rate == 0.0 else nn.Dropout(p=dropout_rate)
        self.bn2 = get_norm(planes, norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(self.lrelu(self.bn1(x))))
        out = self.conv2(self.lrelu(self.bn2(out)))
        out += self.shortcut(x)

        return out


def get_norm(n_filters, norm):
    if norm is None:
        return Identity()
    elif norm == "batch":
        return nn.BatchNorm2d(n_filters, momentum=0.9)
    elif norm == "instance":
        return nn.InstanceNorm2d(n_filters, affine=True)
    elif norm == "layer":
        return nn.GroupNorm(1, n_filters)
    elif norm == "act":
        return norms.ActNorm(n_filters, False)


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=10, input_channels=3,
                 sum_pool=False, norm=None, leak=.2, dropout_rate=0.0):
        super(Wide_ResNet, self).__init__()
        self.leak = leak
        self.in_planes = 16
        self.sum_pool = sum_pool
        self.norm = norm
        self.lrelu = nn.LeakyReLU(leak)

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(input_channels, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)    # 定义第一个block，输入通道数为nStages[0]，输出通道数为nStages[1]
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = get_norm(nStages[3], self.norm)      # bn1的输入输出通道数为nStages[3]，用来归一化
        self.last_dim = nStages[3]      # 定义最后一层的输出维度，用于后面的全连接层
        self.linear = nn.Linear(nStages[3], num_classes)    # 定义全连接层，输入维度为nStages[3]，输出维度为num_classes

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):     # 定义函数用于构建每个block
        strides = [stride] + [1]*(num_blocks-1)              # 第一个block的步长为stride，后面的都为1
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, norm=self.norm))      # 每个block的输入通道数为self.in_planes，输出通道数为planes
            self.in_planes = planes             # 更新self.in_planes为planes

        return nn.Sequential(*layers)       # 返回一个Sequential容器，包含所有的block

    def forward(self, x, vx=None):
        out = self.conv1(x)         # 卷积层
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.lrelu(self.bn1(out))     # 激活层和归一化层
        if self.sum_pool:
            out = out.view(out.size(0), out.size(1), -1).sum(2)     # out结果为[batch_size, 64, 8, 8]，将后两维压缩求和，结果为[batch_size, 64]
        else:
            out = F.avg_pool2d(out, 8)           # out结果为[batch_size, 64, 8, 8]，将后两维压缩求平均，结果为[batch_size, 64, 1, 1]
        out = out.view(out.size(0), -1)         # 将out结果展平为[batch_size, 64]
        return out