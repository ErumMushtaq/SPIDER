import numpy as np
import torch
import torch.nn as nn
import logging
import torch.nn.functional as F

from fedml_api.model.cv.darts.sample_operation import OPS, FactorizedReduce
from ptflops import get_model_complexity_info
# from fedml_api.model.cv.darts.sample_operation import *
import copy
# from operations import *
from torch.autograd import Variable
from fedml_api.model.cv.darts.genotypes import PRIMITIVES, Genotype


# from genotypes import PRIMITIVES
# from genotypes import Genotype


class MixedOp(nn.Module):

    def __init__(self, C, stride, switch, p):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.p = p
        for i in range(len(switch)):
            primitive = PRIMITIVES[i]
            op = OPS[primitive](C, stride, False)
            if switch[i]:
                # primitive = PRIMITIVES[i]
                # op = OPS[primitive](C, stride, False)
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                # if isinstance(op, Identity) and p > 0:
                #     op = nn.Sequential(op, nn.Dropout(self.p))
                # logging.info("New Operation")
                # logging.info(i)
                # logging.info(C)
                # logging.info(stride)
                # logging.info(op)
                self._ops.append(op)

        # print(self.m_ops)

    def update_p(self):
        for op in self.m_ops:
            if isinstance(op, nn.Sequential):
                if isinstance(op[0], Identity):
                    op[1].p = self.p

    def forward(self, x, weights):
        # print("Mixed operation forward")
        # print(weights)
        # print(x)
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, switches, p):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.p = p
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
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
        #     self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        # self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        switch_count = 0
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, switch=switches[switch_count], p=self.p)
                # gives list of all 14 possible connections we can have
                self._ops.append(op)
                switch_count = switch_count + 1

    def update_p(self):
        for op in self._ops:
            op.p = self.p
            op.update_p()

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
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
            s = sum(self._ops[offset + j](h) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class ModelForModelSizeMeasure(nn.Module):
    """
    This class is used only for calculating the size of the generated model.
    The choices of opeartions are made using the current alpha value of the DARTS model.
    The main difference between this model and DARTS model are the following:
        1. The __init__ takes one more parameter "alphas_normal" and "alphas_reduce"
        2. The new Cell module is rewriten to contain the functionality of both Cell and MixedOp
        3. To be more specific, MixedOp is replaced with a fixed choice of operation based on
            the argmax(alpha_values)
        4. The new Cell class is redefined as an Inner Class. The name is the same, so please be
            very careful when you change the code later
        5.
    """

    def __init__(self, C, num_classes, layers, criterion, alphas_normal, alphas_reduce,
                 steps=4, multiplier=4, stem_multiplier=3):
        super(ModelForModelSizeMeasure, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C  # 3*16
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        # for layers = 8, when layer_i = 2, 5, the cell is reduction cell.
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
                cell = InnerCell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,
                                 alphas_reduce)
            else:
                reduction = False
                cell = InnerCell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,
                                 alphas_normal)

            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input_data):
        s0 = s1 = self.stem(input_data)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                s0, s1 = s1, cell(s0, s1)
            else:
                s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


class Network_Global(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, device, args, steps=4, multiplier=4, stem_multiplier=3,
                 switches_normal=[], switches_reduce=[], p=0.0):
        super(Network_Global, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.device = device
        self.p = p
        self.switches_normal = switches_normal
        self.switches_reduce = switches_reduce
        self.any_reduction_layer = False
        switch_ons = []
        for i in range(len(switches_normal)):
            ons = 0
            for j in range(len(switches_normal[i])):
                if switches_normal[i][j]:
                    ons = ons + 1
            switch_ons.append(ons)
            ons = 0
        self.switch_on = switch_ons[0]

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()

        self.k = sum(1 for i in range(self._steps) for n in range(2 + i))
        self.num_ops = len(PRIMITIVES)
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
                self.any_reduction_layer = True
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches_reduce,
                            self.p)
            else:
                reduction = False
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches_normal,
                            self.p)
            #            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        #self._initialize_alphas()

    def switches(self):
        return self.switches_normal, self.switches_reduce

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = torch.ones(self.k, self.num_ops)/self.num_ops
            else:
                weights = torch.ones(self.k, self.num_ops)/self.num_ops
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def update_p(self):
        for cell in self.cells:
            cell.p = self.p
            cell.update_p()

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = self.switch_on
        self.alphas_normal = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(k, num_ops)))
        self.alphas_reduce = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(k, num_ops)))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def update_alphas(self, alphas):
        [alpha_normal, alpha_reduce] = alphas
        logging.info(alpha_normal.shape)
        num_ops = self.switch_on
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        # logging.info("k values "+str(num_ops))
        temp_alphas_normal = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(k, num_ops)))
        temp_alphas_reduce = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(k, num_ops)))
        # logging.info(temp_alphas_normal.shape)
        # logging.info(self.switches_normal)
        for i in range(k):
            idx_normal = 0
            idx_reduce = 0
            for j in range(len(PRIMITIVES)):
                if self.switches_normal[i][j]:
                    with torch.no_grad():
                        # temp_alphas_normal[i][idx_normal] = alpha_normal[i][j]
                        temp_alphas_normal[i][idx_normal] = alpha_normal.data[i][j]
                    idx_normal += 1
                if self.switches_reduce[i][j]:
                    with torch.no_grad():
                        temp_alphas_reduce[i][idx_reduce] = alpha_reduce.data[i][j]
                    idx_reduce += 1

        temp_alphas = [temp_alphas_normal, temp_alphas_reduce]
        return temp_alphas

    def arch_parameters(self):
        return self._arch_parameters

    def get_mask(self):
        return [self.switches_normal, self.switches_reduce]

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

    def get_current_model_size(self, input):
        model = ModelForModelSizeMeasure(self._C, self._num_classes, self._layers, self._criterion,
                                         self.alphas_normal, self.alphas_reduce, self._steps,
                                         4, 3)

        model.to(self.device)
        # logit = model(input)
        # total_flops, total_params = profile(model, inputs=(input,))
        macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
        # logging.info(macs)
        # exit()
        total_params = self.count_parameters()
        total_flops = macs

        # flops = count_ops(model, input)
        # # size = count_parameters_in_MB(model)
        # logging.info('Total Parameter Size {:3f}'.format((total_flops)))
        # logging.info('Total Parameter Size {:3f}'.format((flops)))
        # exit()
        del model
        return total_flops, total_params

    def count_parameters(model):
        return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)
        # return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6



