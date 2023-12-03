import torch
import torch.nn as nn
import logging
OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: nn.Sequential() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    # 'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False),
        nn.Conv2d(C, C, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, groups=C, bias=False),
        nn.Conv2d(C, C, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C, affine=affine),
    ),
    'sep_conv_5x5': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, kernel_size=5, stride=stride, padding=2, groups=C, bias=False),
        nn.Conv2d(C, C, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, kernel_size=5, stride=1, padding=2, groups=C, bias=False),
        nn.Conv2d(C, C, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C, affine=affine),
    ),
    'sep_conv_7x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, kernel_size=7, stride=stride, padding=3, groups=C, bias=False),
        nn.Conv2d(C, C, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, kernel_size=7, stride=1, padding=3, groups=C, bias=False),
        nn.Conv2d(C, C, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C, affine=affine),
    ),
    'dil_conv_3x3': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, kernel_size=3, stride=stride, padding=2, dilation=2,
                  groups=C, bias=False),
        nn.Conv2d(C, C, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C, affine=affine),
    ),
    'dil_conv_5x5': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, kernel_size=5, stride=stride, padding=4, dilation=2,
                  groups=C, bias=False),
        nn.Conv2d(C, C, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C, affine=affine),
    ),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    # DilConv(C, C, 5, stride, 4, 2, affine=affine)
    # nn.Sequential(
    #     nn.ReLU(inplace=False),
    #     nn.Conv2d(C, C, kernel_size=5, stride=stride, padding=4, dilation=2,
    #               groups=C, bias=False),
    #     nn.Conv2d(C, C, kernel_size=1, padding=0, bias=False),
    #     nn.BatchNorm2d(C, affine=affine),
    # )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    # SepConv(C, C, 7, stride, 3, affine=affine)
    # nn.Sequential(
    #     nn.ReLU(inplace=False),
    #     nn.Conv2d(C, C, kernel_size=7, stride=stride, padding=3, groups=C, bias=False),
    #     nn.Conv2d(C, C, kernel_size=1, padding=0, bias=False),
    #     nn.BatchNorm2d(C, affine=affine),
    #     nn.ReLU(inplace=False),
    #     nn.Conv2d(C, C, kernel_size=7, stride=1, padding=3, groups=C, bias=False),
    #     nn.Conv2d(C, C, kernel_size=1, padding=0, bias=False),
    #     nn.BatchNorm2d(C, affine=affine),
    # )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        # logging.info("size of x"+str(x.size()))
        # logging.info("size of x" + str(self.conv_1(x).size()))
        # logging.info("size of x2" + str(self.conv_2(x[:, :, 1:, 1:]).size()))
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
