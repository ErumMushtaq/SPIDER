import logging
import time

import torch
import wandb
from torch import nn
import numpy as np
from fedml_api.distributed.SPIDER.utils import genotype


class SPIDERAggregator(object):

    def __init__(self, train_global, test_global, all_train_data_num, client_num, model, device, args):
        self.train_global = train_global
        self.test_global = test_global

        self.all_train_data_num = all_train_data_num

        self.client_num = client_num
        self.device = device
        self.args = args
        self.model = model

        self.model_dict = dict()
        self.alphas_dict = dict()
        self.alpha_val = dict()
        self.sample_num_dict = dict()
        self.train_acc_dict = dict()
        self.train_loss_dict = dict()
        self.local_acc_dict = dict()
        self.best_local_acc = dict()
        self.flops_dict = dict()
        self.model_size_dict = dict()
        self.reg_loss_dict = dict()
        self.train_global_loss_dict = dict()

        self.train_acc_avg = 0.0
        self.test_acc_avg = 0.0
        self.test_loss_avg = 0.0

        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
            self.best_local_acc[idx] = 0.0

        self.best_accuracy = 0
        self.best_accuracy_different_cnn_counts = dict()
        self.wandb_table = wandb.Table(columns=["Epoch", "Searched Architecture"])

    def get_model(self):
        return self.model

    # def add_local_trained_result(self, index, model_params, alphas, sample_num, train_acc, local_train_loss, global_train_loss,
    #                              reg_loss, local_acc, flops, model_size):
    def add_local_trained_result(self, index, model_params, alphas, alpha_val, sample_num, local_acc, local_train_loss, reg_loss,flops, model_size):
        # logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.local_acc_dict[index] = local_acc  # local_acc


        self.alphas_dict[index] = alphas
        self.alpha_val[index] = alpha_val
        # # logging.info(flops)
        self.flops_dict[index] = flops
        self.model_size_dict[index] = model_size
        self.train_loss_dict[index] = local_train_loss # local loss
        self.reg_loss_dict[index] = reg_loss
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.client_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        averaged_weights = self.__aggregate_weight()
        self.model.load_state_dict(averaged_weights)
        return averaged_weights

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            if self.args.client_sampling is False:
                client_indexes = [client_index for client_index in range(client_num_per_round)]
            else:
                num_clients = min(client_num_per_round, client_num_in_total)
                np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        # logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes


    def __aggregate_weight(self):
        logging.info("################aggregate weights############")
        # start_time = time.time()
        model_list = []
        for idx in range(self.client_num):
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                # logging.info(local_sample_number)
                # logging.info(self.all_train_data_num)
                w = local_sample_number / self.all_train_data_num
                # logging.info(w)
                # exit()
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # clear the memory cost
        model_list.clear()
        del model_list
        self.model_dict.clear()
        # end_time = time.time()
        # logging.info("aggregate weights time cost: %d" % (end_time - start_time))
        return averaged_params



    def statistics(self, round_idx):
        # # # train acc
        # train_acc_list = self.train_acc_dict.values()
        # self.train_acc_avg = sum(train_acc_list) / len(train_acc_list)
        # logging.info('Round {:3d}, Average Train Accuracy {:.3f}'.format(round_idx, self.train_acc_avg))
        # wandb.log({"Train Accuracy": self.train_acc_avg, "round": round_idx})
        # Local train loss
        train_loss_list = self.train_loss_dict.values()
        train_loss_avg = sum(train_loss_list) / len(train_loss_list)
        logging.info('Round {:3d}, Average Local Train Loss {:.3f}'.format(round_idx, train_loss_avg))
        wandb.log({"Train Local Loss": train_loss_avg, "Round": round_idx})
        # # Global Loss
        # train_global_loss_list = self.train_global_loss_dict.values()
        # train_global_loss_avg = sum(train_global_loss_list) / len(train_global_loss_list)
        # logging.info('Round {:3d}, Average Global Train Loss {:.3f}'.format(round_idx, train_global_loss_avg))
        # wandb.log({"Train Global Loss": train_global_loss_avg, "Round": round_idx})
        # # test loss
        # logging.info('Round {:3d}, Average Validation Loss {:.3f}'.format(round_idx, self.test_loss_avg))
        # wandb.log({"Validation Loss": self.test_loss_avg, "Round": round_idx})
        # logging.info("search_train_valid_acc_gap %f" % (self.train_acc_avg - self.test_loss_avg))
        # wandb.log({"search_train_valid_acc_gap": self.train_acc_avg - self.test_loss_avg, "Round": round_idx})
        # Reg loss
        reg_loss_list = self.reg_loss_dict.values()
        reg_loss_list_avg = sum(reg_loss_list) / len(reg_loss_list)
        logging.info('Round {:3d}, Average Reg Loss {:.3f}'.format(round_idx, reg_loss_list_avg))
        wandb.log({"Average Reg Loss": reg_loss_list_avg, "Round": round_idx})

        # Local accuracy
        local_acc_list = self.local_acc_dict.values()
        local_acc_avg = sum(local_acc_list) / len(local_acc_list)
        logging.info('Round {:3d}, Average Local Accuracy {:.3f}'.format(round_idx, local_acc_avg))
        wandb.log({"Average Local Accuracy": local_acc_avg, "round": round_idx})
        # test acc
        logging.info('Round {:3d}, Average Validation Accuracy {:.3f}'.format(round_idx, self.test_acc_avg))
        wandb.log({"Validation Accuracy": self.test_acc_avg, "round": round_idx})

        flops_list = self.flops_dict.values()
        flops_avg = sum(flops_list) / len(flops_list)
        logging.info('Round {:3d}, Average FLOPs {:.3f}'.format(round_idx, flops_avg))
        wandb.log({"Average FLOPs": flops_avg, "Round": round_idx})

        ms_list = self.model_size_dict.values()
        ms_avg = sum(ms_list) / len(ms_list)
        logging.info('Round {:3d}, Average Model Size {:.3f}'.format(round_idx, ms_avg))
        wandb.log({"Average Model Size": ms_avg, "Round": round_idx})


    def record_training_statistics(self, round_idx):
        # Local accuracy
        local_acc_list = self.local_acc_dict.values()
        local_acc_avg = sum(local_acc_list) / len(local_acc_list)
        logging.info('Round {:3d}, Average Local Accuracy {:.3f}'.format(round_idx, local_acc_avg))
        wandb.log({"Average Local Accuracy (Evaluation Phase)": local_acc_avg, "Epoch": round_idx})

        # # # train acc
        # train_acc_list = self.train_acc_dict.values()
        # self.train_acc_avg = sum(train_acc_list) / len(train_acc_list)
        # logging.info('Round {:3d}, Average Baseline {:.3f}'.format(round_idx, self.train_acc_avg))
        # wandb.log({"Baseline Accuracy": self.train_acc_avg, "Epoch": round_idx})

        # for client_idx in range(self.args.client_num_per_round):
        #     local_acc = self.local_acc_dict[client_idx]
        #     logging.info('Round {:3d}, Average Local Accuracy {:.3f}'.format(round_idx, local_acc))
        #     wandb.log({"Average Local Accuracy (Evaluation Phase)_ClientID"+str(client_idx): local_acc, "Epoch": round_idx})

        #     ## Record best local acc as well
        #     if self.local_acc_dict[client_idx] >= self.best_local_acc[client_idx]:
        #         self.best_local_acc[client_idx] = self.local_acc_dict[client_idx]
        #     wandb.log({"Best Local Accuracy (Evaluation Phase)_ClientID" + str(client_idx): self.best_local_acc[client_idx],
        #                    "Epoch": round_idx})

        #     # # train acc
        #     train_acc = self.train_acc_dict[client_idx]
        #     logging.info('Round {:3d}, Average Baseline {:.3f}'.format(round_idx, train_acc))
        #     wandb.log({"Baseline Accuracy_ClientID"+str(client_idx): train_acc, "Epoch": round_idx})

    def infer(self, round_idx):
        self.model.eval()
        self.model.to(self.device)
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            start_time = time.time()
            test_correct = 0.0
            test_loss = 0.0
            test_sample_number = 0.0
            iteration_num = 0
            test_data = self.test_global
            # loss
            criterion = nn.CrossEntropyLoss().to(self.device)
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(test_data):
                    iteration_num += 1
                    x = x.to(self.device)
                    target = target.to(self.device)

                    pred = self.model(x)
                    if self.args.stage == "train":
                        loss = criterion(pred, target)
                        _, predicted = torch.max(pred, 1)
                    else:
                        loss = criterion(pred, target)
                        _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(target).sum()

                    test_correct += correct.item()
                    test_loss += loss.item() * target.size(0)
                    test_sample_number += target.size(0)

                    if iteration_num == 1 and self.args.is_debug_mode:
                        break
                logging.info("server test. round_idx = %d, test_loss = %s" % (round_idx, test_loss))

            self.test_acc_avg = test_correct / test_sample_number
            self.test_loss_avg = test_loss

            end_time = time.time()
            logging.info("server_infer time cost: %d" % (end_time - start_time))

    def record_local_model_architectures(self):
        wandb_table_pers = wandb.Table(columns=["Client Architectures"])
        wandb_table_pers.add_data(str(self.alphas_dict))
        wandb.log({"Client Architectures": wandb_table_pers})

        wandb_table_pers_ = wandb.Table(columns=["Alpha Client Architectures"])
        wandb_table_pers_.add_data(str(self.alpha_val))
        wandb.log({"Alpha Client Architectures": wandb_table_pers_})

    def local_model_statistics(self):
        wandb_table_acc = wandb.Table(columns=["Local Models Accuracies"])
        wandb_table_acc.add_data(str(self.local_acc_dict))
        wandb.log({"Local Models Validation Accuracy": wandb_table_acc})

        wandb_table_best_acc = wandb.Table(columns=["Local Models Accuracies"])
        wandb_table_best_acc.add_data(str(self.best_local_acc))
        wandb.log({"Local Models Best Validation Accuracy": wandb_table_best_acc})

        wandb_table_flops = wandb.Table(columns=["Local Models FLOPs"])
        wandb_table_flops.add_data(str(self.flops_dict))
        wandb.log({"Local Models FLOPs": wandb_table_flops})

        if self.test_acc_avg > self.best_accuracy:
            self.best_accuracy = self.test_acc_avg
            wandb_table_acc = wandb.Table(columns=["Best round Accuracies"])
            wandb_table_acc.add_data(str(self.local_acc_dict))
            wandb.log({"Local Models FLOPs": wandb_table_acc})
            # wandb.run.summary["best_valid_accuracy"] = self.best_accuracy
            # wandb.run.summary["epoch_of_best_accuracy"] = self.round_idx


