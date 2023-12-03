import logging
import os
import shutil

import numpy as np
import pickle
import torch
import wandb
import torch.nn.functional as F
from collections import namedtuple



def transform_list_to_tensor(model_params_list):
    for k in model_params_list.keys():
        model_params_list[k] = torch.from_numpy(np.asarray(model_params_list[k])).float()
    return model_params_list


def transform_tensor_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params


def post_complete_message_to_sweep_process(args):
    os.system("mkdir ./tmp/; touch ./tmp/fedml")
    pipe_path = "./tmp/fedml"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, 'w') as pipe:
        pipe.write("training is finished! \n%s\n" % (str(args)))


def save_checkpoint(model_name, round, model, acc):
    state = {
        'model_name': model_name,
        'round': round,
        'state_dict': model.state_dict(),
        'acc': acc
    }
    folder = "./checkpoint/best_model/"
    filename = "./checkpoint/best_model/" + model_name + "_fednas_best_acc.pth"
    if not os.path.exists(folder):
        os.mkdir(folder)
    torch.save(state, filename)
    #wandb.save(filename)


def load_checkpoint(model, model_name):
    filename = "./checkpoint/best_model/" + model_name + "_fednas_best_acc.pth"
    #checkpoint = torch.load(filename, map_location='cuda:0')
    checkpoint = torch.load(filename, torch.device('cpu'))
    start_round = checkpoint['round']
    acc = checkpoint['acc']
    model.load_state_dict(checkpoint['state_dict'])
    return start_round, model, acc


def clear_cache_for_personalized_model(args):
    folder = args.personalized_model_path + "/personalized_model"
    try:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    except Exception:
        print("failed")


def save_personal_model(args, personalized_model, client_index):
    # create folder
    folder = args.personalized_model_path + "/personalized_model"
    if not os.path.exists(folder):
        os.mkdir(folder)
    path = folder + '/client_' + str(client_index) + '.pth'
    torch.save(personalized_model.cpu().state_dict(), path)  # save the model
    # logging.info(" Personal Model of Client number %d saved " % client_index)

def save_training_model(args, personalized_model, client_index):
    # create folder
    folder = args.personalized_model_path + "/Training_model"
    if not os.path.exists(folder):
        os.mkdir(folder)
    path = folder + '/client_' + str(client_index) + '.pth'
    torch.save(personalized_model.cpu().state_dict(), path)  # save the model
    # logging.info(" Personal Model of Client number %d saved " % client_index)

def save_extra_variables(args, extra_variables, client_index):
    # create folder
    folder = args.personalized_model_path + "/Extra_Variables"
    if not os.path.exists(folder):
        os.mkdir(folder)
    file_path = folder + '/client_' + str(client_index) + '.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(extra_variables, f)
    f.close()

    # torch.save(personalized_model.cpu().state_dict(), path)  # save the model
    logging.info(" Extra Variables of Client number %d saved " % client_index)

def load_extra_variables(args, client_index):
    path = args.personalized_model_path + '/Extra_Variables/client_' + str(client_index) + '.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            varr0, varr1, varr2, varr3 = pickle.load(f)
            logging.info(" Extra Variables of Client number %d Loaded " % client_index)
        f.close()
        return varr0, varr1, varr2, varr3

def load_training_model(args, personalized_model, client_index):
    path = args.personalized_model_path + '/Training_model/client_' + str(client_index) + '.pth'
    if os.path.exists(path):  # checking if there is a file with this name
        personalized_model.load_state_dict(torch.load(path))  # if yes load it
        logging.info(" Personal Model of Client number %d Loaded " % client_index)

def load_personal_model(args, personalized_model, client_index):
    path = args.personalized_model_path + '/personalized_model/client_' + str(client_index) + '.pth'
    if os.path.exists(path):  # checking if there is a file with this name
        personalized_model.load_state_dict(torch.load(path))  # if yes load it
        logging.info(" Personal Model of Client number %d Loaded " % client_index)


def load_personal_train_model(args, personalized_model, client_index):
    path = args.personalized_model_path + '/Train_folder/personalized_model/client_' + str(client_index) + '.pth'
    if os.path.exists(path):  # checking if there is a file with this name
        personalized_model.load_state_dict(torch.load(path))  # if yes load it
        logging.info(" Personal Model of Client number %d Loaded " % client_index)

def load_train_checkpoint(model, model_name):
    filename = "./checkpoint/Train_folder/best_model/" + model_name + "_fednas_best_acc.pth"
    #checkpoint = torch.load(filename, map_location='cuda:0')
    checkpoint = torch.load(filename, torch.device('cpu'))
    start_round = checkpoint['round']
    acc = checkpoint['acc']
    model.load_state_dict(checkpoint['state_dict'])
    return start_round, model, acc

def compare_models(weights1, weights2):
    models_differ = 0
    logging.info("Compare Model Called")
    for key_item_1, key_item_2 in zip(weights1.items(), weights2.items()):
        #if key_item_2[1].dtype == torch.float32:
        if torch.equal(key_item_1[1], key_item_2[1]):
            # logging.info(key_item_1[1])
            # logging.info(key_item_2[1])
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                logging.info('Mismtach found at', key_item_1[0])
            else:
                raise Exception

    if models_differ == 0:
        logging.info('Models match perfectly! :)')
    logging.info(models_differ)


# To print architectures
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


PRIMITIVES = [
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'skip_connect',
    'max_pool_3x3',
    'avg_pool_3x3',
    'none',
]

def genotype(alphas_normal, alphas_reduce):
    _steps = 4
    _multiplier = 4

    def _isCNNStructure(k_best):
        return k_best >= 4

    def _parse(weights):
        _steps = 4
        gene = []
        n = 2
        start = 0
        cnn_structure_count = 0
        _multiplier = 4
        for i in range(_steps):
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
        gene_normal, cnn_structure_count_normal = _parse(F.softmax(alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce, cnn_structure_count_reduce = _parse(F.softmax(alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + _steps - _multiplier, _steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
    return genotype, cnn_structure_count_normal, cnn_structure_count_reduce
