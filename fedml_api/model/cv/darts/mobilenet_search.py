import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fedml_api.model.cv.darts.genotypes import PRIMITIVES, Genotype
from fedml_api.model.cv.darts.operations import OPS, FactorizedReduce, ReLUConvBN


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        # w is the operation mixing weights. see equation 2 in the original paper.
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
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
        # s0 and s1 has shape [64, 48, 14, 14])

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class InnerCell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, weights):
        super(InnerCell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        # len(self._ops)=2+3+4+5=14
        offset = 0
        keys = list(OPS.keys())
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                weight = weights.data[offset + j]
                choice = keys[weight.argmax()]
                op = OPS[choice](C, stride, False)
                if 'pool' in choice:
                    op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                self._ops.append(op)
            offset += i + 2

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            # s2 = self._ops[0 + 0](s0) + self._ops[1 + 1](s1)
            # states = [s0, s1, s2]
            # offset = 2

            # s3 =

            # s4

            # s5

            s = sum(self._ops[offset + j](h) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        # [s2, s3, s4, s5]
        return torch.cat(states[-self._multiplier:], dim=1)


class MobileNet_Network(nn.Module):

    def __init__(self, backbone_model, batch_size, C, num_classes, layers, criterion, device, steps=4, multiplier=4,
                 stem_multiplier=3):
        super(MobileNet_Network, self).__init__()
        print(MobileNet_Network)
        self.backbone_model = backbone_model
        self.batch_size = batch_size
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._stem_multiplier = stem_multiplier


        self.device = device

        C_curr = stem_multiplier * C  # 3*16
        # checking mobilenet
        C_in = 2560
        #C_in = self.backbone_model.config.hidden_size

        # padding = 1
        #if self.design_number == 4:
        self.stem = nn.Sequential(
                nn.Conv2d(C_in, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr)
            )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False

            # for layers = 8, when layer_i = 2, 5, the cell is reduction cell.
        for i in range(layers):
            if i == 0:
                logging.info("create one layer normal cell")
                reduction = False
            # else:
            #     if i in [layers // 3, 2 * layers // 3]:
            #         C_curr *= 2
            #         reduction = True
            #     else:
                    #reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

            self.global_pooling = nn.AdaptiveAvgPool2d(1)
            #logging.info("Model search C_in" + str(C_prev))
            self.classifier = nn.Linear(64, num_classes)
            #self.classifier = nn.Linear(C_prev, num_classes)

            self._initialize_alphas()

        #if self.design_number == 2:
        #    self.classifier = nn.Linear( C_in*196, num_classes)  # change 196 later


    def new(self):
        model_new = Network(self.backbone_model, self.batch_size, self._C, self._num_classes,
                            self._layers, self._criterion, self.device).to(self.device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        #logging.info("Size of Input " + str(input.size()))
        mixed_hidden_feature = self.backbone_model(input) #[64, 768, 196]
        s0 = s1 = self.stem(mixed_hidden_feature)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        concat_vect = torch.cat([out.view(out.size(0), -1)], 1) #[64, 832]
        logits = self.classifier(concat_vect)  # C_prev, number of classes
        return logits

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i)) # 14
        num_ops = len(PRIMITIVES) # 8, may be the number of functions we have


        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops)) # tensor of dimension [8x14]
        self._arch_parameters = [
            self.alphas_normal,
        ]


    def new_arch_parameters(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i)) # 14
        num_ops = len(PRIMITIVES) # 8

        alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops)).to(self.device) # [8x14]
        _arch_parameters = [
            alphas_normal,
        ]

        return _arch_parameters

    def arch_parameters(self):
        logging.info(" Size of Arch Parameters "+str(self._arch_parameters))
        return self._arch_parameters

    def get_pi_parameters(self):
        return self._scalar_parameters

    def genotype(self):

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

                # select the largest two connections
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]

                # select the operation which has the largest P.
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k

                    if _isCNNStructure(k_best):
                        cnn_structure_count += 1

                    # output format: (best operation, connection index)
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene, cnn_structure_count

        with torch.no_grad():
            gene_normal, cnn_structure_count_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())

            concat = range(2 + self._steps - self._multiplier, self._steps + 2)
            genotype = Genotype(
                normal=gene_normal, normal_concat=concat,
                reduce=None, reduce_concat=None
            )
        return genotype, cnn_structure_count_normal
