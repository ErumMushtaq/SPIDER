import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from thop import profile
import numpy as np
import copy
from pthflops import count_ops

# from darts.genotypes import PRIMITIVES, Genotype
# from darts.operations import OPS, FactorizedReduce, ReLUConvBN
# from darts.utils import count_parameters_in_MB

from fedml_api.model.cv.darts.genotypes import PRIMITIVES, Genotype
from fedml_api.model.cv.darts.sample_operation import OPS, FactorizedReduce, ReLUConvBN
#from fedml_api.model.cv.darts.operations import OPS, FactorizedReduce, ReLUConvBN


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        #self._ops = nn.ModuleList()
        self._ops = nn.ModuleDict()
        self.ops_list = dict()
        index = 0
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.add_module(str(primitive), op)
            self.ops_list[index] = (str(primitive))
            index = index + 1
            #self._ops.append(op)

    def forward(self, x, weights, cpu_weights):
        clist = []
        #logging.info(cpu_weights)
        mydict = self._ops.keys()
        logging.info(mydict)

        for j, cpu_weight in enumerate(cpu_weights):
            name_ops = self.ops_list[j]
            if j == 2:
                cpu_weight = 0
            if abs(cpu_weight) != 0:
                clist.append(weights[j] * self._ops[name_ops](x))
            elif name_ops in self._ops.keys(): # delete the operation if exists
                self._ops.pop(name_ops)

        # logging.info(self._ops.keys())
        # exit()
        #
        # for j, cpu_weight in enumerate(cpu_weights):
        #     if abs(cpu_weight) > 1e-10:
        #         name_ops = self.ops_list[j]
        #         clist.append(weights[j] * self._ops[name_ops](x))
        if len(clist) == 1:
            return clist[0]
        else:
            return sum(clist)
        # w is the operation mixing weights. see equation 2 in the original paper.
        #return sum(w * op(x) for w, op in zip(weights, self._ops))
        #self.weight.data = self.weight.data * self.mask.data

        # for w, op in zip(weights, self._ops):
        #     logging.info(op(x))
        #     exit()
        #     new_weight = w*op

        # return sum(w * op(x) for w, op in zip(weights, self._ops))




class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
            # self.preprocess0 = nn.Sequential(
            #     nn.ReLU(inplace=False),
            #     nn.Conv2d(C_prev_prev, C, 1, stride=1, padding=0, bias=False),
            #     nn.BatchNorm2d(C, affine=False)
            # )
        # self.preprocess1 = nn.Sequential(
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(C_prev, C, 1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(C, affine=False)
        # )
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
        cpu_weights = weights.tolist()

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(
                self._ops[offset + j](h, weights[offset + j], cpu_weights[offset + j]) for j, h in enumerate(states))
            #s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
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


class Network2(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, device, args, steps=4, multiplier=4, stem_multiplier=3):
        super(Network2, self).__init__()
        # print(Network)
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._stem_multiplier = stem_multiplier
        self._args = args

        self.device = device

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
            if i in [layers // 3, 2 * layers // 3] and i > 2 :
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.initialize_mask()

        self._initialize_alphas()
        self.topk = 0
        #self.update_mask()
        #self.topk = 0


    def new(self):
        model_new = Network2(self._C, self._num_classes, self._layers, self._criterion, self.device).to(self.device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        # device_num = input.get_device()
        # #mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        # device = 'cuda:'+str(device_num)
        # self.reduction_mask.to(device)
        # self.normal_mask.to(device)
        # logging.info()
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                # logging.info(self.alphas_reduce)
                #self.alphas_reduce.data = (self.alphas_reduce * self.reduction_mask)
                #logging.info(self.alphas_reduce)
                # exit()
                weights = F.softmax(self.alphas_reduce, dim=-1)
                weights2 = (weights * self.reduction_mask)
                normalize = weights2.sum(1, keepdim=True)
                weights = weights2 / normalize
            else:
                #self.alphas_normal.data = (self.alphas_normal * self.normal_mask)
                #logging.info(self.alphas_normal)
                weights = F.softmax(self.alphas_normal, dim=-1)
                #weights = F.softmax(self.alphas_reduce, dim=-1)
                weights2 = (weights * self.normal_mask)
                normalize = weights2.sum(1, keepdim=True)
                weights = weights2 / normalize
                # weights = (weights * self.normal_mask)
            s0, s1 = s1, cell(s0, s1, weights)
            # logging.info("Workshop Paper Model Search")
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def return_topk(self):
        return self.topk

    def get_min_k(self, input_in, k):
        input = copy.deepcopy(input_in)
        index = []
        # logging.info(input)
        for i in range(k):
            idx = np.argmin(input)
            index.append(idx)
            input[idx] = 1
        # logging.info(index)
        return index

    def update_mask(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        logging.info("Before")
        logging.info(self.normal_mask)
        logging.info(self.reduction_mask)
        # normal_prob = F.softmax(self.alphas_normal, dim=-1).data
        # reduction_prob = F.softmax(self.alphas_reduce, dim=-1).data

        temp_alphas_normal = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(k, num_ops)))
        temp_alphas_reduce = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(k, num_ops)))

        for i in range(k):
            idx_normal = 0
            idx_reduce = 0
            for j in range(len(PRIMITIVES)):
                with torch.no_grad():
                    # temp_alphas_normal[i][idx_normal] = alpha_normal[i][j]
                    temp_alphas_normal[i][idx_normal] = self.alphas_normal.data[i][j]
                    temp_alphas_reduce[i][idx_reduce] = self.alphas_reduce.data[i][j]
                idx_reduce += 1

        # normal_prob = F.softmax(self.alphas_normal, dim=-1).data.numpy()
        # reduction_prob = F.softmax(self.alphas_reduce, dim=-1).data.numpy()
        with torch.no_grad():
            normal_prob = F.softmax(temp_alphas_normal, dim=-1).data.cpu().numpy()
            weights2 = (normal_prob * self.normal_mask.data.cpu().numpy())
            normalize = np.sum(weights2, axis=-1).reshape((14, 1))
            normal_prob = weights2 / normalize
            reduction_prob = F.softmax(temp_alphas_reduce, dim=-1).data.cpu().numpy()
            weights2 = (reduction_prob * self.reduction_mask.data.cpu().numpy())
            normalize = np.sum(weights2, axis=-1).reshape((14, 1))
            reduction_prob = weights2 / normalize
            if self.topk <= 1:
                self.topk = self.topk + 1 # topk connections to drop
                for i in range(k):
                    idxs = []
                    for j in range(num_ops):
                        idxs.append(j)
                        drop_normal = self.get_min_k(normal_prob[i, :], self.topk)
                        drop_reduce = self.get_min_k(reduction_prob[i, :], self.topk)
                    for idx in drop_normal:
                        self.normal_mask[i][idxs[idx]] = 0
                    for idx in drop_reduce:
                        self.reduction_mask[i][idxs[idx]] = 0

            logging.info("After")
            logging.info(self.normal_mask)
            logging.info(self.reduction_mask)

    def get_mask(self):
        return [self.normal_mask, self.reduction_mask]

    def initialize_mask(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        self.normal_mask = torch.ones(k, num_ops).to(self.device)
        self.reduction_mask = torch.ones(k, num_ops).to(self.device)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        if self._args.layers > 2:
            self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
            self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
            self._arch_parameters = [
                self.alphas_normal,
                self.alphas_reduce,
            ]
        else:
            self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
            self._arch_parameters = [
                self.alphas_normal
            ]

    def new_arch_parameters(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        if self._args.layers > 2:
            alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops)).to(self.device)
            alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops)).to(self.device)
            _arch_parameters = [
                alphas_normal,
                alphas_reduce,
            ]
        else:
            alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops)).to(self.device)
            _arch_parameters = [
                alphas_normal
            ]

        return _arch_parameters

    def arch_parameters(self):
        return self._arch_parameters

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
            normal_weights = F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy()
            weights2 = (normal_weights * self.normal_mask.data.cpu().numpy())
            normalize = np.sum(weights2, axis=-1).reshape((14, 1))
            #normalize = weights2.sum(1)
            normal_weights = weights2 / normalize
            gene_normal, cnn_structure_count_normal = _parse(normal_weights)
            #gene_normal, cnn_structure_count_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
            if self._args.layers > 2:
                reduce_weights = F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy()
                weights2 = (reduce_weights * self.reduction_mask.data.cpu().numpy())
                #logging.info(weights2)
                #normalize = weights2.sum(1, keepdim=True)
                normalize = np.sum(weights2, axis=-1).reshape((14, 1))
                reduction_weights = weights2 / normalize
                gene_reduce, cnn_structure_count_reduce = _parse(reduction_weights)
                #gene_reduce, cnn_structure_count_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

            concat = range(2 + self._steps - self._multiplier, self._steps + 2)

            if self._args.layers > 2:
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
                                         self._multiplier, self._stem_multiplier)
        model.to(self.device)
        # logit = model(input)
        total_flops, total_params = profile(model, inputs=(input,))



        # flops = count_ops(model, input)
        # # size = count_parameters_in_MB(model)
        # logging.info('Total Parameter Size {:3f}'.format((total_flops)))
        # logging.info('Total Parameter Size {:3f}'.format((flops)))
        # exit()
        del model
        return total_flops, total_params




def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6