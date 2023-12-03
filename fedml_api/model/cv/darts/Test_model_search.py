import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from fedml_api.model.cv.darts.genotypes import PRIMITIVES, Genotype
# from fedml_api.model.cv.darts.operations import OPS, FactorizedReduce, ReLUConvBN
from fedml_api.model.cv.darts.sample_operation import OPS, FactorizedReduce


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights, cpu_weights):
        clist = []
        max_idx = np.argmax(cpu_weights)
        clist.append(weights[max_idx] * self._ops[int(max_idx)](x))
        # for j, cpu_weight in enumerate(cpu_weights):
        #     if abs(cpu_weight) > 1e-10:
        #         clist.append(weights[j] * self._ops[j](x))
        if len(clist) == 1:
            return clist[0]
        else:
            return sum(clist)


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            # self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
            # (self, C_in, C_out, kernel_size, stride, padding, affine=True)
            # self.op = nn.Sequential(
            #     nn.ReLU(inplace=False),
            #     nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            #     nn.BatchNorm2d(C_out, affine=affine)
            # )
            self.preprocess0 = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_prev_prev, C, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(C, affine=False)
            )
        # self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_prev, C, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C, affine=False)
        )
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        cpu_weights = weights.tolist()
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(
                self._ops[offset + j](h, weights[offset + j], cpu_weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        # logging.info(states)
        return torch.cat(states[-self._multiplier:], dim=1)


class Network_GumbelSoftmax(nn.Module):

    # def __init__(self, C, num_classes, layers, criterion, device, steps=4, multiplier=4, stem_multiplier=3):
    def __init__(self, backbone_model, batch_size, C, num_classes, layers, criterion, device, args, steps=4,
                 multiplier=4,
                 stem_multiplier=3):
        super(Network_GumbelSoftmax, self).__init__()
        self.backbone_model = backbone_model
        self.batch_size = batch_size
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.device = device
        self.args = args
        self.any_reduction_layer = False

        C_curr = stem_multiplier * C  # 3*16
        if self.args.model == 'resnet18':
            C_in = 512
        elif self.args.model == 'efficientnet':
            C_in = 2304
        else:
            C_in = 3
        # C_in = 3
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        # for layers = 8, when layer_i = 2, 5, the cell is reduction cell.
        for i in range(layers):
            if (self._layers > 2) and (i == self._layers - 2):
                # if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
                self.any_reduction_layer = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.tau = 5

        self._initialize_alphas()

    def new(self):
        model_new = Network_GumbelSoftmax(self._C, self._num_classes, self._layers, self._criterion, self.device).to(
            self.device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def set_tau(self, tau):
        self.tau = tau

    def get_tau(self):
        return self.tau

    def forward(self, input):
        if self.args.model == 'resnet18':
            output_last_layer = self.backbone_model(input)  # [B, 512, 7, 7]
            dimensions = list(output_last_layer.shape)
            hidden_feature = torch.reshape(output_last_layer, (dimensions[0], -1, 1, 1))
            logging.info("mixed_hidden_feature.shape = %s" % str(hidden_feature.shape))
            s0 = s1 = self.stem(hidden_feature)

            # logging.debug("output_last_layer.shape = %s" % str(output_last_layer.shape))
            # s0 = s1 = self.stem(output_last_layer)
            # logging.debug("s0.shape = %s" % str(s0.shape))
            # logging.debug("s1.shape = %s" % str(s1.shape))
        elif self.args.model == 'efficientnet':
            mixed_hidden_feature = self.backbone_model(input)  # [B, 1280, 8, 8]
            # dimensions = list(mixed_hidden_feature.shape)
            # hidden_feature = torch.reshape(mixed_hidden_feature, (dimensions[0], -1, 7, 7))
            # logging.info("mixed_hidden_feature.shape = %s" % str(hidden_feature))
            s0 = s1 = self.stem(mixed_hidden_feature)
        else:
            mixed_hidden_feature = self.backbone_model(input)  # [B, 1280, 8, 8]
            logging.info("mixed_hidden_feature.shape = %s" % str(mixed_hidden_feature.shape))
            s0 = s1 = self.stem(input)

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.gumbel_softmax(self.alphas_reduce, self.tau, True)
            else:
                weights = F.gumbel_softmax(self.alphas_normal, self.tau, True)
            s0, s1 = s1, cell(s0, s1, weights)
            logging.info(s0.shape)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        if self.any_reduction_layer is True:
            self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
            self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
            self._arch_parameters = [
                self.alphas_normal,
                self.alphas_reduce,
            ]
        else:
            self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
            self._arch_parameters = [
                self.alphas_normal,
            ]

    def arch_parameters(self):
        return self._arch_parameters

    def base_arch_weight_parameters(self):
        return self.backbone_model.state_dict()

    def get_base_arch_flops(self, device):
        input = torch.randn(1, 3, 224, 224)
        input = input.to(device)
        flops, params = profile(self.backbone_model, inputs=(input,))
        return flops, params

    def genotype(self):
        print(self.alphas_normal.shape)
        print(self.alphas_reduce.shape)

        def _isCNNStructure(k_best):
            return k_best >= 4

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            cnn_structure_count = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k

                    if _isCNNStructure(k_best):
                        cnn_structure_count += 1
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene, cnn_structure_count

        with torch.no_grad():
            gene_normal, cnn_structure_count_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
            gene_reduce, cnn_structure_count_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

            concat = range(2 + self._steps - self._multiplier, self._steps + 2)
            if self.any_reduction_layer is True:
                genotype = Genotype(
                    normal=gene_normal, normal_concat=concat,
                    reduce=gene_reduce, reduce_concat=concat
                )
            else:
                genotype = Genotype(
                    normal=gene_normal, normal_concat=concat
                )

        return genotype, cnn_structure_count_normal, cnn_structure_count_reduce
