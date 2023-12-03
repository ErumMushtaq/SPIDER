import logging
import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from thop import profile
from torch import nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from pthflops import count_ops

from fedml_api.model.cv.darts.utils import KL_Loss
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_api.model.cv.resnet_fednas.resnet import resnet18

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


def up_transform():
    CIFAR_MEAN = [0.5, 0.5, 0.5]
    CIFAR_STD = [0.5, 0.5, 0.5]
    up_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return up_transform

class Architect(object):

    def __init__(self, model, criterion, args, device, teacher_model):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.criterion = criterion
        self.args = args
       # self.KL_loss = KL_Loss(self.args.temperature)
        self.teacher_model = teacher_model
        self.up_transform = up_transform()

        #self._resnet18 = resnet18()

        arch_parameters = self.model.arch_parameters()
        # logging.info(arch_parameters)
        # alpha optimize: Adam
        self.optimizer = torch.optim.Adam(
            arch_parameters,
            lr=args.arch_learning_rate, betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay)

        self.device = device

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        logits = self.model(input)
        loss = self.criterion(logits, target)

        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model

    # DARTS
    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            # logging.info("first order")
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            # logging.info("second order")
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    # FedNAS workshop version
    def step_v222(self, input_train, target_train, input_valid, target_valid, lambda_train_regularizer,
                lambda_valid_regularizer):
        self.optimizer.zero_grad()

        # grads_alpha_with_train_dataset
        logits = self.model(input_train)
        loss_train = self.criterion(logits, target_train)

        arch_parameters = self.model.arch_parameters()
        grads_alpha_with_train_dataset = torch.autograd.grad(loss_train, arch_parameters)
        # grads_alpha_with_train_dataset = torch.autograd.grad(loss_train, arch_parameters, allow_unused=True)

        self.optimizer.zero_grad()

        # grads_alpha_with_val_dataset
        logits = self.model(input_valid)
        loss_val = self.criterion(logits, target_valid)

        arch_parameters = self.model.arch_parameters()
        grads_alpha_with_val_dataset = torch.autograd.grad(loss_val, arch_parameters)


        for g_train, g_val in zip(grads_alpha_with_train_dataset, grads_alpha_with_val_dataset):
            temp = g_train.data.mul(lambda_train_regularizer)
            g_val.data.add_(temp)

        arch_parameters = self.model.arch_parameters()
        for v, g in zip(arch_parameters, grads_alpha_with_val_dataset):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
        self.optimizer.step()
        #logging.info("Gradient taken")

    # FedNAS workshop version for SPIDER
    def step_v2222(self, input_train, target_train, input_valid, target_valid, lambda_train_regularizer,
                lambda_valid_regularizer, global_weiights, local_weights, lambda_):
        self.optimizer.zero_grad()

        # grads_alpha_with_train_dataset
        logits = self.model(input_train)
        loss_train = self.criterion(logits, target_train)

        arch_parameters = self.model.arch_parameters()
        grads_alpha_with_train_dataset = torch.autograd.grad(loss_train, arch_parameters)
        # grads_alpha_with_train_dataset = torch.autograd.grad(loss_train, arch_parameters, allow_unused=True)

        self.optimizer.zero_grad()

        # grads_alpha_with_val_dataset
        logits = self.model(input_valid)
        loss_val = self.criterion(logits, target_valid)

        arch_parameters = self.model.arch_parameters()
        grads_alpha_with_val_dataset = torch.autograd.grad(loss_val, arch_parameters)


        for g_train, g_val in zip(grads_alpha_with_train_dataset, grads_alpha_with_val_dataset):
            temp = g_train.data.mul(lambda_train_regularizer)
            g_val.data.add_(temp)

        arch_parameters = self.model.arch_parameters()
        for v, g in zip(arch_parameters, grads_alpha_with_val_dataset):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer.step()

    # ours
    def step_v2(self, input_train, target_train):
        # logging.info("step v2")
        start_time = time.time()
        self.optimizer.zero_grad()
        self.model.to(self.device)
        # grads_alpha_with_train_dataset
        logits = self.model(input_train)
        if self.args.beta > 0:
            """
            calculate the efficiency loss term
            """
            input = torch.randn(1, 3, 32, 32)
            input = input.to(self.device)
            #TODO: number of parameters look v small
            total_flops, total_params = self.model.get_current_model_size(input)
            # logging.info(
            #     'FedNAS workshop Cell: Parameter Size {:3f}, Total # of flops {:.3f}'.format((total_params / 1e6), (total_flops / 1e6)))
            # factor_params = 0.3
            #efficiency_loss = (1 - factor_params) * (total_flops / 1e9) + factor_params * (total_params / 1e6)
            #efficiency_loss = (total_flops / 1e9) #+ factor_params * (total_params / 1e6)
            efficiency_loss = (total_flops / 1e8)
            # Values: ~0.16
            # logging.info("Efficiency Loss value %f" % efficiency_loss)
        else:
             efficiency_loss = 0.0

        if self.args.gamma > 0:
            """
                    calculate the KL loss term
            """
            student_logits = self.model(input_train)
            teacher_logits = self.teacher_model(input_train).detach()
            KL_loss_train = self.KL_loss(student_logits, teacher_logits)
        else:
            KL_loss_train = 0.0

        loss_train = self.criterion(logits, target_train)
        # logging.info("Before ")
        # arch_parameters = self.model.arch_parameters()
        # logging.info(arch_parameters)
        # logging.info(
        #         'CE Loss {:3f}, KL loss {:.3f}, Efficiency Loss {:.3f} for alphas'.format((loss_train), (KL_loss_train),
        #                                                                                    (efficiency_loss)))
        #
        # logging.info(
        #         'CE Loss {:3f}, KL loss*gamma {:.3f}, Efficiency Loss*beta {:.3f} for alphas'.format((loss_train), (self.args.gamma* KL_loss_train),
        #                                                                                    (self.args.beta* efficiency_loss)))

        # logging.info(loss_train)
        # logging.info(efficiency_loss)
        loss_train = loss_train + self.args.beta * efficiency_loss + self.args.gamma * KL_loss_train
        # logging.info(loss_train)
        arch_parameters = self.model.arch_parameters()
        grads_alpha_with_train_dataset = torch.autograd.grad(loss_train, arch_parameters)
        self.optimizer.zero_grad()

        arch_parameters = self.model.arch_parameters()
        for v, g in zip(arch_parameters, grads_alpha_with_train_dataset):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
        nn.utils.clip_grad_norm_(arch_parameters, self.args.grad_clip)
        self.optimizer.step()
        # logging.info("After ")
        # arch_parameters = self.model.arch_parameters()
        # logging.info(arch_parameters)
        # end_time = time.time()
        # logging.info(start_time)
        # logging.info(end_time)
        # logging.info("Total time cost: %d" % (end_time - start_time))
        # exit()

    # ours
    def step_per(self, input_train, target_train, gamma, beta, device):
        """
        calculate the efficiency loss term
        """
        input = torch.randn(1, 3, 224, 224)
        input = input.to(device)

        total_flops, total_params = profile(self.model, inputs=(input,))
        logging.debug(
            'Total Parameter Size {:3f}, Total # of flops {:.3f}'.format((total_params / 1e6), (total_flops / 1e9)))

        base_flops, base_params = self.model.get_base_arch_flops(device)
        logging.debug(
            'Base Parameter Size {:3f}, # of base flops {:.3f}'.format((base_params / 1e6), (base_flops / 1e9)))

        cell_flops = total_flops - base_flops
        cell_params = total_params - base_params
        logging.debug(
            'cell Parameter Size {:3f}, # of cell flops {:.3f}'.format((cell_params / 1e6), (cell_flops / 1e9)))
        factor_params = 0.3
        efficiency_loss = (1 - factor_params) * (cell_flops / 1e9) + factor_params * (cell_params / 1e6)
        logging.debug("Efficiency Loss value %f" % efficiency_loss)

        """
        calculate the CE loss term
        """
        logits = self.model(input_train)
        loss_train = self.criterion(logits, target_train) + beta * efficiency_loss

        """
        calculate the KL loss term
        """
        logits_1 = self.model(input_train)
        teacher_logits = self.teacher_model(input_train).detach()
        KL_loss_train = self.KL_loss(teacher_logits, logits_1)
        loss_train += gamma * KL_loss_train
        logging.info(
            'CE Loss {:3f}, KL loss {:.3f}, Efficiency Loss {:.3f} for alphas'.format((loss_train), (KL_loss_train),
                                                                                      (efficiency_loss)))

        self.optimizer.zero_grad()
        arch_parameters = self.model.arch_parameters()
        grads_alpha_with_train_dataset = torch.autograd.grad(loss_train, arch_parameters)
        for v, g in zip(arch_parameters, grads_alpha_with_train_dataset):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
        nn.utils.clip_grad_norm_(arch_parameters, self.args.grad_clip)
        self.optimizer.step()

    # ours
    def step_single_level(self, input_train, target_train):
        self.optimizer.zero_grad()

        # grads_alpha_with_train_dataset
        logits = self.model(input_train)
        loss_train = self.criterion(logits, target_train)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        grads_alpha_with_train_dataset = torch.autograd.grad(loss_train, arch_parameters)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        for v, g in zip(arch_parameters, grads_alpha_with_train_dataset):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer.step()

    def step_wa(self, input_train, target_train, input_valid, target_valid, lambda_regularizer):
        self.optimizer.zero_grad()

        # grads_alpha_with_train_dataset
        logits = self.model(input_train)
        loss_train = self.criterion(logits, target_train)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        grads_alpha_with_train_dataset = torch.autograd.grad(loss_train, arch_parameters)

        # grads_alpha_with_val_dataset
        logits = self.model(input_valid)
        loss_val = self.criterion(logits, target_valid)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        grads_alpha_with_val_dataset = torch.autograd.grad(loss_val, arch_parameters)

        for g_train, g_val in zip(grads_alpha_with_train_dataset, grads_alpha_with_val_dataset):
            temp = g_train.data.mul(lambda_regularizer)
            g_val.data.add_(temp)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        for v, g in zip(arch_parameters, grads_alpha_with_val_dataset):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer.step()

    def step_AOS(self, input_train, target_train, input_valid, target_valid):
        self.optimizer.zero_grad()
        output_search = self.model(input_valid)
        arch_loss = self.criterion(output_search, target_valid)
        arch_loss.backward()
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        logits = self.model(input_valid)
        loss = self.criterion(logits, target_valid)

        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        # calculate w' in equation (7):
        # approximate w(*) by adapting w using only a single training step and enable momentum.
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)

        logits = unrolled_model(input_valid)
        unrolled_loss = self.criterion(logits, target_valid)
        unrolled_loss.backward()  # w, alpha

        # the first term of equation (7)
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]

        # vector is (gradient of w' on validation dataset)
        vector = [v.grad.data for v in unrolled_model.parameters()]

        # equation (8) = 2nd term of equation (7)
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        # equation (7)
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()

        for v, g in zip(arch_parameters, dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        named_parameters = self.model.module.named_parameters() if self.is_multi_gpu else self.model.named_parameters()
        for k, v in named_parameters:
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)

        if self.is_multi_gpu:
            new_state_dict = OrderedDict()
            for k, v in model_dict.items():
                logging.info("multi-gpu")
                logging.info("k = %s, v = %s" % (k, v))
                if 'module' not in k:
                    k = 'module.' + k
                else:
                    k = k.replace('features.module.', 'module.features.')
                new_state_dict[k] = v
        else:
            new_state_dict = model_dict

        model_new.load_state_dict(new_state_dict)
        return model_new.to(self.device)

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        # vector is (gradient of w' on validation dataset)
        R = r / _concat(vector).norm()
        parameters = self.model.module.parameters() if self.is_multi_gpu else self.model.parameters()
        for p, v in zip(parameters, vector):
            p.data.add_(R, v)  # w+ in equation (8) # inplace operation

        # get alpha gradient based on w+ in training dataset
        logits = self.model(input)
        loss = self.criterion(logits, target)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        grads_p = torch.autograd.grad(loss, arch_parameters)

        parameters = self.model.module.parameters() if self.is_multi_gpu else self.model.parameters()
        for p, v in zip(parameters, vector):
            p.data.sub_(2 * R, v)  # w- in equation (8)

        # get alpha gradient based on w- in training dataset
        logits = self.model(input)
        loss = self.criterion(logits, target)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        grads_n = torch.autograd.grad(loss, arch_parameters)

        # restore w- to w
        parameters = self.model.module.parameters() if self.is_multi_gpu else self.model.parameters()
        for p, v in zip(parameters, vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    # DARTS
    def step_v2_2ndorder(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer,
                         lambda_train_regularizer, lambda_valid_regularizer):
        self.optimizer.zero_grad()

        # approximate w(*) by adapting w using only a single training step and enable momentum.
        # w has been updated to w'
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        # print("BEFORE:" + str(unrolled_model.parameters()))

        """(7)"""
        logits_val = unrolled_model(input_valid)
        valid_loss = self.criterion(logits_val, target_valid)
        valid_loss.backward()  # w, alpha

        # the 1st term of equation (7)
        grad_alpha_wrt_val_on_w_prime = [v.grad for v in unrolled_model.arch_parameters()]

        # vector is (gradient of w' on validation dataset)
        grad_w_wrt_val_on_w_prime = [v.grad.data for v in unrolled_model.parameters()]

        # the 2nd term of equation (7)
        implicit_grads = self._hessian_vector_product(grad_w_wrt_val_on_w_prime, input_train, target_train)

        # equation (7)
        for g, ig in zip(grad_alpha_wrt_val_on_w_prime, implicit_grads):
            g.data.sub_(eta, ig.data)

        grad_alpha_term = unrolled_model.new_arch_parameters()
        for g_new, g in zip(grad_alpha_term, grad_alpha_wrt_val_on_w_prime):
            g_new.data.copy_(g.data)

        """(8)"""
        # unrolled_model_train = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_model.zero_grad()

        logits_train = unrolled_model(input_train)
        train_loss = self.criterion(logits_train, target_train)
        train_loss.backward()  # w, alpha

        # the 1st term of equation (8)
        grad_alpha_wrt_train_on_w_prime = [v.grad for v in unrolled_model.arch_parameters()]

        # vector is (gradient of w' on validation dataset)
        grad_w_wrt_train_on_w_prime = [v.grad.data for v in unrolled_model.parameters()]

        # the 2nd term of equation (7)
        implicit_grads = self._hessian_vector_product(grad_w_wrt_train_on_w_prime, input_train, target_train)

        # equation (7)
        for g, ig in zip(grad_alpha_wrt_train_on_w_prime, implicit_grads):
            g.data.sub_(eta, ig.data)

        for g_train, g_val in zip(grad_alpha_wrt_train_on_w_prime, grad_alpha_term):
            # g_val.data.copy_(lambda_valid_regularizer * g_val.data)
            # g_val.data.add_(g_train.data.mul(lambda_train_regularizer))
            temp = g_train.data.mul(lambda_train_regularizer)
            g_val.data.add_(temp)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        for v, g in zip(arch_parameters, grad_alpha_term):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer.step()

    # DARTS
    def step_v2_2ndorder2(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer,
                          lambda_train_regularizer, lambda_valid_regularizer):
        self.optimizer.zero_grad()

        # approximate w(*) by adapting w using only a single training step and enable momentum.
        # w has been updated to w'
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        # print("BEFORE:" + str(unrolled_model.parameters()))

        """(7)"""
        logits_val = unrolled_model(input_valid)
        valid_loss = self.criterion(logits_val, target_valid)
        valid_loss.backward()  # w, alpha

        # the 1st term of equation (7)
        grad_alpha_wrt_val_on_w_prime = [v.grad for v in unrolled_model.arch_parameters()]

        # vector is (gradient of w' on validation dataset)
        grad_w_wrt_val_on_w_prime = [v.grad.data for v in unrolled_model.parameters()]

        # the 2nd term of equation (7)
        implicit_grads = self._hessian_vector_product(grad_w_wrt_val_on_w_prime, input_valid, target_valid)

        # equation (7)
        for g, ig in zip(grad_alpha_wrt_val_on_w_prime, implicit_grads):
            g.data.sub_(eta, ig.data)

        grad_alpha_term = unrolled_model.new_arch_parameters()
        for g_new, g in zip(grad_alpha_term, grad_alpha_wrt_val_on_w_prime):
            g_new.data.copy_(g.data)

        """(8)"""
        # unrolled_model_train = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_model.zero_grad()

        logits_train = unrolled_model(input_train)
        train_loss = self.criterion(logits_train, target_train)
        train_loss.backward()  # w, alpha

        # the 1st term of equation (8)
        grad_alpha_wrt_train_on_w_prime = [v.grad for v in unrolled_model.arch_parameters()]

        # vector is (gradient of w' on validation dataset)
        grad_w_wrt_train_on_w_prime = [v.grad.data for v in unrolled_model.parameters()]

        # the 2nd term of equation (7)
        implicit_grads = self._hessian_vector_product(grad_w_wrt_train_on_w_prime, input_train, target_train)

        # equation (7)
        for g, ig in zip(grad_alpha_wrt_train_on_w_prime, implicit_grads):
            g.data.sub_(eta, ig.data)

        for g_train, g_val in zip(grad_alpha_wrt_train_on_w_prime, grad_alpha_term):
            g_val.data.copy_(lambda_valid_regularizer * g_val.data)
            g_val.data.add_(g_train.data.mul(lambda_train_regularizer))

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        for v, g in zip(arch_parameters, grad_alpha_term):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer.step()

# # if self.args.stage == 'personalized_search' or self.args.stage == 'backbone_cell_search': # if combined with base cell
# base_flops, base_params = profile(self._resnet18, inputs=(input,))
# logging.info(
#         'ResNet18: Parameter Size {:3f}, # of base flops {:.3f}'.format((base_params / 1e6), (base_flops / 1e6)))

# total_flops = total_flops - base_flops
# total_params = total_params - base_params
# logging.info(
#         'cell Parameter Size {:3f}, # of cell flops {:.3f}'.format((total_params / 1e6), (total_flops / 1e6)))

