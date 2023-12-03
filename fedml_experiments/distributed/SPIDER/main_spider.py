import argparse
import logging
import os
import socket
import sys
import random

import numpy as np
import psutil
import setproctitle
import torch
import wandb
# https://nyu-cds.github.io/python-mpi/05-collectives/
from mpi4py import MPI
# add the FedML root directory to the python path
from torch import nn
from torchinfo import summary

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_api.model.cv.darts.projection import pt_project
from fedml_api.model.cv.darts.darts_proj import DartsNetworkProj

from fedml_api.data_preprocessing.cifar100.ICLR_data_loader import load_partition_data_cifar100
#from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_distributed_cifar100
from fedml_api.distributed.SPIDER.SPIDERAPI import FedML_init, FedML_SPIDER_distributed
from fedml_api.distributed.SPIDER.utils import load_personal_model, load_personal_train_model,compare_models, load_checkpoint
from fedml_api.model.cv.darts import genotypes
from fedml_api.model.cv.darts.model import NetworkCIFAR
from fedml_api.model.cv.darts.model_search_workshop_code import Network2
from fedml_api.model.cv.darts.model_train_workshop_code import Network_Train
from fedml_api.model.cv.darts.architect_ig import Architect
import fedml_api.model.cv.darts.utils as ig_utils
from fedml_api.model.cv.darts.genotypes import PRIMITIVES, Genotype
from fedml_api.model.cv.darts.model_search_pdarts import Network
from fedml_api.model.cv.darts.model_search_pdarts_code import Network_Global
from fedml_api.model.cv.darts.model_search_rethink import Network as DartsNetwork
from fedml_api.model.cv.darts.spaces import spaces_dict
from fedml_api.model.cv.darts.Rethink_model_search.global_model_search import Global_Network
#from fedml_api.model.cv.darts.Rethink_model_search.model_search import Local_Network
from fedml_api.model.cv.darts.Rethink_model_search.projection_model_search import DartsNetworkProj
from fedml_api.model.cv.darts.Rethink_model_search.Projection_global_model_search import GDartsNetworkProj
from ptflops import get_model_complexity_info
from fedml_api.model.cv.darts.model_search import FedNASNetwork

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--gpu_server_num', type=int, default=1,help='gpu_server_num')
    parser.add_argument('--gpu_num_per_server', type=int, default=4, help='gpu_num_per_server')
    parser.add_argument('--stage', type=str, default='search', help='stage: search; train')
    parser.add_argument('--model', type=str, default='resnet', metavar='N', help='neural network used in training')

    # for data distribution
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N', help='dataset used for training')
    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10', help='data directory')
    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N', help='how to partition the dataset on local workers')
    parser.add_argument('--partition_alpha', type=float, default=0.2, metavar='PA', help='partition alpha (default: 0.5)')
    parser.add_argument('--classes_per_client', type=int, default=5, help='debug mode')

    # Training HPs
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=5, metavar='EP', help='how many epochs will be trained locally')
    parser.add_argument('--local_points', type=int, default=5000, metavar='LP', help='the approximate fixed number of data points we will have on each local worker')
    parser.add_argument('--client_number', type=int, default=4, metavar='NN', help='number of workers in a distributed cluster')
    parser.add_argument('--seed', type=int, default=9, metavar='NN', help='number of workers in a distributed cluster')
    parser.add_argument('--client_num_per_round', type=int, default=4, metavar='NN', help='number of workers in a distributed cluster') #for client sampling
    parser.add_argument('--local_finetune', type=str, default=True, help='local fine_tune')
    parser.add_argument('--loss', type=str, default='mix', help='CE KL mix')
    parser.add_argument('--temperature', type=float, default=1, help='temperature parameter for KL loss')
    parser.add_argument('--client_num_in_total', type=int, default=100, metavar='NN',   help='number of workers in a distributed clusters (added)')
    parser.add_argument('--comm_round', type=int, default=50,help='how many round of communications we shoud use')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--class_num', type=int, default=10, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='DARTS layers')

    #parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--local_lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001)')

    # parser.add_argument("--pretrained_dir", type=str,
    #                     default="./../../../fedml_api/model/cv/pretrained/Transformer/vit/ViT-B_16.npz",
    #                     help="Where to search for pretrained vit models.")


    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')

    #parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    # parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
    #                     help='how to partition the dataset on local workers')

    # parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    # parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--lambda_train_regularizer', type=float, default=1, help='train regularizer parameter')
    parser.add_argument('--lambda_valid_regularizer', type=float, default=1, help='validation regularizer parameter')
    parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
    parser.add_argument('--tau_max', type=float, default=10, help='initial tau')
    parser.add_argument('--tau_min', type=float, default=1, help='minimum tau')
    parser.add_argument('--fednas_type', type=str, default='hetero', help='Hetero or Homo (old)')
    parser.add_argument('--is_debug_mode', type=int, default=0, help='debug mode')
    parser.add_argument("--img_size", default=32, type=int, help="Resolution size")
    parser.add_argument('--client_sampling', action='store_true', default=False, help='use auxiliary tower')


    # parser.add_argument('--perturb_alpha', type=str, default='none', help='perturb for alpha')
    # parser.add_argument('--epsilon_alpha', type=float, default=0.3, help='max epsilon for alpha')
    # parser.add_argument('--method', type=str, default='darts', help='darts, darts-proj, sdarts, sdarts-proj')
    ## projection
    parser.add_argument('--edge_decision', type=str, default='random', choices=['random'],help='used for both proj_op and proj_edge')
    parser.add_argument('--proj_crit_normal', type=str, default='acc', choices=['loss', 'acc'])
    parser.add_argument('--proj_crit_reduce', type=str, default='acc', choices=['loss', 'acc'])
    parser.add_argument('--proj_crit_edge', type=str, default='acc', choices=['loss', 'acc'])
    parser.add_argument('--proj_intv', type=int, default=1, help='interval between two projections')
    parser.add_argument('--proj_start', type=int, default=60, help='first round of starting projections')
    parser.add_argument('--proj_recovery', type=int, default=30, help='interval between projections')
    parser.add_argument('--proj_mode_edge', type=str, default='reg', choices=['reg'], help='edge projection evaluation mode, reg: one edge at a time')
    parser.add_argument('--tune_epochs', type=int, default=140, help='not used for projection (use proj_intv instead)')
    parser.add_argument('--fast', action='store_true', default=False, help='eval/train on one batch, for debugging')
    parser.add_argument('--dev_resume_epoch', type=int, default=-1,help="resume epoch for arch selection phase, starting from 0")
    parser.add_argument('--dev_resume_log', type=str, default='', help="resume log name for arch selection phase")
    parser.add_argument('--dev_resume_checkpoint_dir', type=str, default="./checkpoint/proj", help="personalization_method: None; pFedMe; ditto; perFedAvg")
    parser.add_argument('--dev_save_checkpoint_dir', type=str, default="./checkpoint/proj_save", help="personalization_method: None; pFedMe; ditto; perFedAvg")
    parser.add_argument('--FL', type=str, default='False', choices=['False', 'True'])
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--arch', type=str, default='FedNAS_V1', help='which architecture to use')
    parser.add_argument('--frequency_of_the_test', type=int, default=10, help='the frequency of the test')
    #parser.add_argument('--frequency_of_the_test', type=int, default=2, help='the frequency of the test')

    # parser.add_argument('--gamma', type=float, default=0,
    #                     help='gamma value for KL loss')
    parser.add_argument('--fednas_design', type=int, default=3)
    parser.add_argument('--gpu_starting', type=int, default=4,help='1: FedAVG, 2: Ditto, 3: Proposed')
    # parser.add_argument('--beta', type=float, default=0,
    #                     help='beta value for efficiency loss')
    parser.add_argument('--run_id', type=int, default=0)

    # Step 2
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='accumulation_steps')
    parser.add_argument('--client_optimizer', type=str, default='sgd', help='SGD with momentum; adam')
    parser.add_argument('--pssl_lambda', type=float, default=2, help="personalization_method: None; pFedMe; "
                                                                     "ditto; perFedAvg")
    parser.add_argument('--personalized_model_path', type=str, default="./checkpoint",
                        help="personalization_method: None; pFedMe; ditto; perFedAvg")

    args = parser.parse_args()
    return args


def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    if process_ID == 0:
        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
        return device
    process_gpu_dict = dict()
    for client_index in range(fl_worker_num):
        gpu_index = (client_index % gpu_num_per_machine) + args.gpu_starting
        process_gpu_dict[client_index] = gpu_index

    logging.info(process_gpu_dict)
    device = torch.device("cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    logging.info(device)
    return device

if __name__ == "__main__":
    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    parser = argparse.ArgumentParser()
    args = add_args(parser)
    str_process_name = "SPIDER:" + str(rank)

    setproctitle.setproctitle(str_process_name)

    logging.basicConfig(level=logging.INFO,
                        format=str(rank) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(rank) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            # project="federated_nas",
            project="SPIDER",
            name="SPIDER" + str(args.partition_method) + "r" + str(args.comm_round) + "-e" + str(
                args.epochs)+ "-lr"+str(args.lr) + "-lambda "+str(args.pssl_lambda)+"-stage"+str(args.stage)+str(args.loss),
            config=args
        )

    seed = 9
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Therefore, we can see that workers are assigned according to the order of machine list.
    logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server)

    # load data
    if args.dataset == "cifar100":
        args.data_dir = './../../../data/cifar100'
        data_loader = load_partition_data_cifar100
        # dataset, data_dir, partition_method, partition_alpha, client_number, batch_size
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, valid_data_local_dict, test_data_local_dict, \
        class_num, contrain_data_local_dict = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_number,
                                args.batch_size)

    args.class_num = class_num

    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #Create directories to save the models
    if process_id == 0:
        if not os.path.exists(args.personalized_model_path + "_" + str(args.partition_method) + "_" + str(args.partition_alpha) + "_" + str(args.dataset)+ "_"+ str(args.run_id)):
            os.mkdir(args.personalized_model_path + "_" + str(args.partition_method) + "_" + str(args.partition_alpha) + "_" + str(args.dataset)+"_"+ str(args.run_id))
        if args.stage == 'search':
            # delete all files
            folder = args.personalized_model_path + "_" + str(args.partition_method) + "_" + str(args.partition_alpha) + "_" + str(args.dataset) + "_"+ str(args.run_id)+ "/personalized_model/"
            if os.path.exists(folder):
                filelist = [f for f in os.listdir(folder)]
                logging.info(" Deleting the following files ")
                logging.info(filelist)
                if len(filelist) != 0:
                    for f in filelist:
                        os.remove(os.path.join(folder, f))
            else:
                os.mkdir(folder)

            folder = args.personalized_model_path + "_" + str(args.partition_method) + "_" + str(args.partition_alpha) + "_" + str(args.dataset) +"_"+ str(args.run_id)+ "/best_model/"
            if os.path.exists(folder):
                filelist = [f for f in os.listdir(folder)]
                logging.info(" Deleting the following files ")
                logging.info(filelist)
                if len(filelist) != 0:
                    for f in filelist:
                        os.remove(os.path.join(folder, f))
            else:
                os.mkdir(folder)

        folder = args.dev_resume_checkpoint_dir
        if os.path.exists(folder):
            filelist = [f for f in os.listdir(folder)]
            logging.info(" Deleting the following files ")
            logging.info(filelist)
            if len(filelist) != 0:
                for f in filelist:
                    os.remove(os.path.join(folder, f))
        else:
            os.mkdir(folder)

        folder = args.dev_save_checkpoint_dir
        if os.path.exists(folder):
            filelist = [f for f in os.listdir(folder)]
            logging.info(" Deleting the following files ")
            logging.info(filelist)
            if len(filelist) != 0:
                for f in filelist:
                    os.remove(os.path.join(folder, f))
        else:
            os.mkdir(folder)

    # create model
    model = None
    criterion = nn.CrossEntropyLoss().to(device)

    logging.info('Initializing Model')
    logging.info(device)
    local_model = GDartsNetworkProj(args.init_channels, class_num, args.layers, criterion, spaces_dict['s2'], args, device) #s5
    global_model = GDartsNetworkProj(args.init_channels, class_num, args.layers, criterion, spaces_dict['s2'], args,
                                device)

    logging.info('Model Initialization complete')
    FedML_SPIDER_distributed(process_id, worker_number, device, comm,
                             global_model, local_model, train_data_num, train_data_global, test_data_global,
                             train_data_local_num_dict, train_data_local_dict, valid_data_local_dict, test_data_local_dict, contrain_data_local_dict, args)
