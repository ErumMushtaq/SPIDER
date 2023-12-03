import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fedml_api.model.cv.darts.sample_operation import *
import copy
# from operations import *
from torch.autograd import Variable
from fedml_api.model.cv.darts.genotypes import PRIMITIVES, Genotype


# from genotypes import PRIMITIVES
# from genotypes import Genotype


class MixedOp(nn.Module):

    def __init__(self, C, stride, switch, p):
        super(MixedOp, self).__init__()
        self.m_ops = nn.ModuleList()
        self.p = p
        for i in range(len(switch)):
            primitive = PRIMITIVES[i]
            op = OPS[primitive](C, stride, False)
            if switch[i]:
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                # if isinstance(op, Identity) and p > 0:
                #     op = nn.Sequential(op, nn.Dropout(self.p))
                self.m_ops.append(op)


    def forward(self, x, weights, cpu_weights):
        clist = []
        max_idx = np.argmax(cpu_weights)
        clist.append(weights[max_idx] * self.m_ops[int(max_idx)](x))
        # for j, cpu_weight in enumerate(cpu_weights):
        #     if abs(cpu_weight) > 1e-10:
        #         clist.append(weights[j] * self._ops[j](x))
        if len(clist) == 1:
            return clist[0]
        else:
            return sum(clist)
        #return sum(w * op(x) for w, op in zip(weights, self.m_ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, switches, p):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.p = p
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self.cell_ops = nn.ModuleList()
        switch_count = 0
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, switch=switches[switch_count], p=self.p)
                # gives list of all 14 possible connections we can have
                self.cell_ops.append(op)
                switch_count = switch_count + 1

    def update_p(self):
        for op in self.cell_ops:
            op.p = self.p
            op.update_p()

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        cpu_weights = weights.tolist()
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(
                self.cell_ops[offset + j](h, weights[offset + j], cpu_weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, device, args, steps=4, multiplier=4, stem_multiplier=3,
                 switches_normal=[], switches_reduce=[], p=0.0):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
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

        self._initialize_alphas()
        self.tau = 5

    def switches(self):
        return self.switches_normal, self.switches_reduce

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                if self.alphas_reduce.size(1) == 1:
                    weights = F.softmax(self.alphas_reduce, dim=0)
                else:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                if self.alphas_normal.size(1) == 1:
                    weights = F.softmax(self.alphas_normal, dim=0)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

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
        logging.info("k values " + str(num_ops))
        temp_alphas_normal = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(k, num_ops)))
        temp_alphas_reduce = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(k, num_ops)))
        logging.info(temp_alphas_normal.shape)
        logging.info(self.switches_normal)
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



