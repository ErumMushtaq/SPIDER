import logging

import torch
from torch import nn
import math
import numpy as np
import copy
import torch.nn.functional as F
import time
from fedml_api.model.cv.darts import utils
# from fedml_api.model.cv.darts.architect import Architect
from ptflops import get_model_complexity_info
import fedml_api.model.cv.darts.utils as ig_utils
from fedml_api.model.cv.darts.spaces import spaces_dict

# from fedml_api.distributed.ICLR_fednas.DittoOpt import DittoSSLOptimizer
from fedml_api.distributed.SPIDER.utils2 import compare_models, load_personal_model, load_train_checkpoint, save_extra_variables, load_extra_variables, save_personal_model, save_checkpoint, save_training_model, load_training_model, load_checkpoint
from fedml_api.model.cv.darts.utils import KL_Loss
from fedml_api.model.cv.darts.model_search_pdarts import Network
from fedml_api.model.cv.darts.projection import project_op, project_edge
from fedml_api.model.cv.darts.model import NetworkCIFAR
from fedml_api.model.cv.darts.Rethink_model_search.projection_model_search import DartsNetworkProj
from fedml_api.model.cv.darts.Rethink_model_search.Projection_global_model_search import GDartsNetworkProj

class SPIDERTrainer(object):
    def __init__(self, client_index, train_local, valid_local, test_local, local_sample_number, all_train_data_num, global_model, local_model, device,
                 contrain_data_local_dict,args):
        logging.info('Trainer Initialized ')
        self.client_index = client_index
        self.train_local_dict = train_local
        self.test_local_dict = test_local
        self.valid_local_dict = valid_local
        self.local_sample_number_dict = local_sample_number
        self.contrain_data_local_dict = contrain_data_local_dict
        self.args = args
        self.train_local = train_local[client_index]
        self.test_local = test_local[client_index]
        self.valid_local = valid_local[client_index]
        self.local_sample_number = local_sample_number[client_index]
        self.global_model_copy = copy.deepcopy(global_model)


        self.all_train_data_num = all_train_data_num
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temp_global_model = GDartsNetworkProj(args.init_channels, args.class_num, args.layers, self.criterion, spaces_dict['s2'], args, device)
        self.args.proj_crit = {'normal': self.args.proj_crit_normal, 'reduce': self.args.proj_crit_reduce}
        self.global_model = global_model
        self.local_model = local_model
        self.pruned_edges = 0
        self.projected_edges = 0
        self.global_model.to(self.device)
        self.local_model.to(self.device)
        self.temp_global_model.to(self.device)
        self.round_index = 0.0
        self.best_acc = 0.0
        self.topk = 0


        # Initialize Optimizers
        arch_parameters = self.local_model.arch_parameters()
        arch_params = list(map(id, arch_parameters))
        parameters = self.local_model.parameters()
        weight_params = filter(lambda p: id(p) not in arch_params, parameters)
        global_arch_parameters = self.global_model.arch_parameters()
        global_arch_params = list(map(id, global_arch_parameters))
        global_parameters = self.global_model.parameters()
        global_weight_params = filter(lambda p: id(p) not in global_arch_params, global_parameters)
        self.g_optimizer = torch.optim.SGD(global_weight_params, lr=self.args.lr, momentum=self.args.momentum, weight_decay=0.001)
        self.l_optimizer = torch.optim.SGD(weight_params, lr=self.args.local_lr, momentum=self.args.momentum, weight_decay=0.001)
        self.g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.g_optimizer, float(self.args.comm_round), eta_min=self.args.learning_rate_min)
        self.l_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.l_optimizer, float(self.args.comm_round), eta_min=self.args.learning_rate_min)


    def update_model(self, weights):
        # logging.info("update_model. client_index = %d" % self.client_index)
        # if self.round_index == 30:
        #     self.local_model.load_state_dict(weights)
        self.global_model.load_state_dict(weights)
        # self.local_model.load_state_dict(weights, False)
        self.global_model_copy.load_state_dict(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_local_dict[client_index]
        self.test_local = self.test_local_dict[client_index]
        self.valid_local = self.valid_local_dict[client_index]
        self.local_sample_number = self.local_sample_number_dict[client_index]
        if self.pruned_edges >= 14:
            # self.args.pssl_lambda = 1
            self.train_local = self.contrain_data_local_dict[client_index]


    def update_training_progress(self, round_index):
        self.round_index = round_index

    def adjust_learning_rate(self, optimizer, initial_lr, round_index, total_round):
        """Decay the learning rate based on schedule"""
        lr = initial_lr
        # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * round_index / total_round))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def count_parameters(self):
        return sum(p.numel() for p in self.local_model.parameters() if p.requires_grad)

    # local search
    def search(self):

        load_personal_model(self.args, self.local_model, self.client_index)
        # training
        self.local_model = self.local_model.to(self.device)
        self.global_model = self.global_model.to(self.device)
        self.temp_global_model.to(self.device)
        self.local_model.train()
        self.global_model.train()

        self.adjust_learning_rate(self.g_optimizer, self.args.lr, self.round_index, self.args.comm_round)
        self.adjust_learning_rate(self.l_optimizer, self.args.local_lr, self.round_index, self.args.comm_round)

        local_avg_train_acc = []
        local_avg_local_train_loss = []
        local_avg_global_train_loss = []
        local_avg_reg_loss = []

        if  self.round_index % self.args.proj_recovery == 0 and self.round_index >= self.args.proj_start and self.pruned_edges <= 13:
            logging.info("Prunning")
            selected_eid_normal, best_opid_normal = project_op(self.local_model, self.valid_local, self.args, self.device, infer,
                                                            cell_type='normal')
            self.local_model.project_op(selected_eid_normal, best_opid_normal, cell_type='normal')
            self.temp_global_model.project_op(selected_eid_normal, best_opid_normal, cell_type='normal')
            selected_eid_reduce, best_opid_reduce = project_op(self.local_model, self.valid_local, self.args, self.device, infer,
                                                            cell_type='reduce')
            self.local_model.project_op(selected_eid_reduce, best_opid_reduce, cell_type='reduce')
            self.temp_global_model.project_op(selected_eid_reduce, best_opid_reduce, cell_type='reduce')
            self.pruned_edges += 1

            for batch_idx, (input_, target) in enumerate(self.train_local):
                input_ = input_.to(self.device)
                logits = self.local_model(input_, update_operations=1)
                logits = self.temp_global_model(input_, update_operations=1)
                # break

        if self.round_index % self.args.proj_recovery == 0 and self.round_index >= self.args.proj_start and self.pruned_edges >= 14 and self.projected_edges <=2:
            logging.info("Prunning")
            selected_nid_normal, eids_normal = project_edge(self.local_model, self.valid_local, self.args, self.device, infer, cell_type='normal')
            self.local_model.project_edge(selected_nid_normal, eids_normal, cell_type='normal')
            self.temp_global_model.project_edge(selected_nid_normal, eids_normal, cell_type='normal')
            selected_nid_reduce, eids_reduce = project_edge(self.local_model, self.valid_local, self.args, self.device, infer, cell_type='reduce')
            self.local_model.project_edge(selected_nid_reduce, eids_reduce, cell_type='reduce')
            self.temp_global_model.project_edge(selected_nid_reduce, eids_reduce, cell_type='reduce')
            if self.projected_edges == 2:
                var_1, var2, var3, var4 = self.local_model.return_proj_weights()
                save_extra_variables(self.args, [var_1, var2, var3, var4], self.client_index)
                #Testing reloading part as well
                varr0, varr1, varr2, varr3 = load_extra_variables(self.args, self.client_index)
                logging.info(varr0)
            self.projected_edges += 1

            for batch_idx, (input_, target) in enumerate(self.train_local):
                input_ = input_.to(self.device)
                logits = self.local_model(input_, update_operations=1)
                logits = self.temp_global_model(input_, update_operations=1)
                # break

        for epoch in range(self.args.epochs):
            train_acc, train_obj, local_train_loss, global_train_loss, reg_loss = self.local_search_design3(self.train_local, self.test_local)
            local_avg_train_acc.append(train_acc)
            local_avg_local_train_loss.append(local_train_loss)
            local_avg_global_train_loss.append(global_train_loss)
            local_avg_reg_loss.append(reg_loss)


        weights = self.global_model.cpu().state_dict() # report global weights for fedavg
        arch = self.local_model.cpu().genotype()

        alpha_val = []
        alpha_val.append(self.local_model.get_projected_weights(cell_type='normal'))
        alpha_val.append(self.local_model.get_projected_weights(cell_type='reduce'))
        save_personal_model(self.args, self.local_model, self.client_index)

        if self.round_index % self.args.frequency_of_the_test == 0:
            with torch.no_grad(): 
                acc_local_model_on_local_data, valid_loss = self.local_infer_duplicate(self.test_local, self.local_model,self.criterion, '')
                macs, params = get_model_complexity_info(self.local_model, (3, 32, 32), as_strings=False, print_per_layer_stat=False, verbose=False)

            if acc_local_model_on_local_data >= self.best_acc:
                self.best_acc = acc_local_model_on_local_data
                save_checkpoint(str(self.client_index), self.round_index, self.local_model, acc_local_model_on_local_data)

        else:
            acc_local_model_on_local_data = 0.0
            macs = 0.0
            params = 0.0

        return weights, arch, alpha_val, self.local_sample_number, acc_local_model_on_local_data,  self.client_index, sum(local_avg_local_train_loss) / len(local_avg_local_train_loss), sum(local_avg_reg_loss) / len(local_avg_reg_loss), macs, params

    def local_search_design3(self, train_queue, valid_queue):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        self.global_model.to(self.device)
        self.local_model.to(self.device)
        loss = None
        iteration_num = 0
        batch_loss = []
        x_accumulator = []
        epoch_loss = []
        self.local_model.train()
        self.global_model.train()

        weights1_ = copy.deepcopy(self.temp_global_model.state_dict())

        global_weights = self.global_model_copy.state_dict()
        self.temp_global_model.load_state_dict(global_weights, strict=False)
        weights2_ = copy.deepcopy(self.temp_global_model.state_dict())

        # Filter out alpha parameters of temp parameters for regularization
        temp_arch_parameters = self.temp_global_model.arch_parameters()
        temp_arch_params = list(map(id, temp_arch_parameters))
        temp_parameters = self.temp_global_model.parameters() # have alpha learning params
        temp_weight_params = filter(lambda p: id(p) not in temp_arch_params,
                            temp_parameters)

        global_model_param = [p for p in temp_weight_params if p.requires_grad]
                        # Local model parameters
        local_arch_parameters = self.local_model.arch_parameters()
        local_arch_params = list(map(id, local_arch_parameters))
        reg_loss_ = []
        local_loss_ = []
        for batch_idx, (input_, target) in enumerate(self.train_local):
            n = input_.size(0)
            iteration_num +=1

            self.g_optimizer.zero_grad()

            input_ = input_.to(self.device)
            target = target.to(self.device)
            logits = self.global_model(input_)
            global_loss = self.criterion(logits, target)
            global_loss.backward()
            global_parameters = self.global_model.parameters()
            nn.utils.clip_grad_norm_(global_parameters, self.args.grad_clip)
            # prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            self.g_optimizer.step()
            self.g_optimizer.zero_grad()
            global_loss = global_loss.cpu()
 
            # local model optimization
            self.l_optimizer.zero_grad()
            logits_ = self.local_model(input_)
            local_loss = self.criterion(logits_, target)
            local_loss.backward()

            local_parameters = self.local_model.parameters()  # have alpha learning params

            local_weight_params = filter(lambda p: id(p) not in local_arch_params,
                                        local_parameters)
            local_model_param = [p for p in local_weight_params if p.requires_grad]

            reg_loss = 0.0

            for (p, g_p) in zip(local_model_param, global_model_param):
            # for (p, g_p) in zip(local_parameters, temp_parameters):
                reg_loss += (self.args.pssl_lambda * 0.5) * torch.linalg.norm(p - g_p.data) ** 2
            reg_loss.backward()
            
            nn.utils.clip_grad_norm_(local_parameters, self.args.grad_clip)
            self.l_optimizer.step()
            self.l_optimizer.zero_grad()
            # if self.round_index > 1:
            reg_loss = reg_loss.cpu()
            reg_loss_.append(reg_loss.item()/n)
            # else:
            #     reg_loss_.append(0)
            local_loss_.append(local_loss.item()/n)
            
            if batch_idx % self.args.report_freq == 0:
                logging.info('client_index = %d, local loss %f, reg loss %f, global loss %f ', self.client_index,
                            local_loss, reg_loss, global_loss)


            if iteration_num == 1 and self.args.is_debug_mode:
                break


        return 0.0, 0.0, np.mean(local_loss_), global_loss, np.mean(reg_loss_)


    # def local_infer(self, valid_queue, model, criterion, type):
    #     objs = utils.AvgrageMeter()
    #     top1 = utils.AvgrageMeter()
    #     top5 = utils.AvgrageMeter()
    #     model.to(self.device)
    #     model.eval()
    #     loss = None
    #     iteration_num = 0
    #     test_correct = 0
    #     test_loss = 0
    #     test_sample_number = 0
    #     for step, (input, target) in enumerate(valid_queue):
    #         logging.info(target)
    #         iteration_num += 1
    #         input = input.to(self.device)
    #         target = target.to(self.device)

    #         if type == 'student':
    #             logits, logits_aux = model(input)
    #         else:
    #             logits = model(input)
    #         loss = criterion(logits, target)
    #         prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    #         n = input.size(0)
    #         objs.update(loss.item(), n)
    #         top1.update(prec1.item(), n)
    #         top5.update(prec5.item(), n)


    #         if iteration_num == 1 and self.args.is_debug_mode:
    #             break

    #     return top1.avg / 100.0, objs.avg / 100.0, loss

    # after searching, infer() function is used to infer the searched architecture
    def local_infer_duplicate(self, valid_queue, model, criterion, type):
        model.to(self.device)
        model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        iteration_num = 0
        #test_data = self.train_local
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(valid_queue):
                iteration_num += 1
                x = x.to(self.device)
                target = target.to(self.device)
                logits = model(x, weights_dict=None)

                loss = criterion(logits, target)
                _, predicted = torch.max(logits, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)

                if iteration_num == 1 and self.args.is_debug_mode:
                    break

            logging.info("client_idx = %d, local_infer_loss = %s" % (self.client_index, test_loss))
        return test_correct / test_sample_number, test_loss
    # after searching, infer() function is used to infer the searched architecture
    def infer(self, valid_queue, model, criterion, type_):
        model.to(self.device)
        model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        iteration_num = 0
        #test_data = self.train_local
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(valid_queue):
                iteration_num += 1
                x = x.to(self.device)
                target = target.to(self.device)

                if type_ == 'student':
                    logits = model(x)
                    # logits, logits_aux = model(x)
                else:
                    logits = model(x)

                loss = criterion(logits, target)
                _, predicted = torch.max(logits, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)

                if iteration_num == 1 and self.args.is_debug_mode:
                    break

            logging.info("client_idx = %d, local_infer_loss = %s" % (self.client_index, test_loss))
        return test_correct / test_sample_number, test_loss
def infer(valid_queue, model, device, args, log=True, _eval=True, weights_dict=None):
    objs = ig_utils.AvgrageMeter()
    top1 = ig_utils.AvgrageMeter()
    top5 = ig_utils.AvgrageMeter()
    model.eval() if _eval else model.train()  # disable running stats for projection

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.to(device)
            target = target.to(device, non_blocking=True)

            if weights_dict is None:
                logits = model(input, weights_dict=None)
                loss = model._criterion(logits, target)
            else:
                logits = model(input, weights_dict=weights_dict)
                loss = model._criterion(logits, target)

            prec1, prec5 = ig_utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if args.is_debug_mode == 1:
                break

    return top1.avg, objs.avg

