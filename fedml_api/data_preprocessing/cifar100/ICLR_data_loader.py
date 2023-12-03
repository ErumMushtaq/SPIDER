import logging

import numpy as np
import torch
import random
import seaborn as sns
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import CIFAR100_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _plot_label_distribution(client_number, class_num, net_dataidx_map, y_complete):
    heat_map_data = np.zeros((class_num, client_number))

    for client_idx in range(client_number):
        idxx = net_dataidx_map[client_idx]
        logging.info("idxx = %s" % str(idxx))
        logging.info("y_train[idxx] = %s" % y_complete[idxx])

        valuess, counts = np.unique(y_complete[idxx], return_counts=True)
        logging.info("valuess = %s" % valuess)
        logging.info("counts = %s" % counts)
        # exit()
        for (i, j) in zip(valuess, counts):
            heat_map_data[i][int(client_idx)] = j / len(idxx)

    # data_dir = args.figure_path
    # fig_name = " cifar100+ "_%s_clients_heatmap_label.png" % args.partition_name"
    # fig_dir = os.path.join(data_dir, fig_name)
    plt.figure()
    fig_dims = (30, 10)
    # fig, ax = plt.subplots(figsize=fig_dims)
    # sns.set(font_scale=4)
    sns.heatmap(heat_map_data, linewidths=0.05, cmap="YlGnBu", cbar=True)
    plt.xlabel('Client number')
    # plt.ylabel('ratio of the specific label data w.r.t total dataset')
    # ax.tick_params(labelbottom=False, labelleft=False, labeltop=False, left=False, bottom=False, top=False)
    # fig.tight_layout(pad=0.1)
    plt.title("label distribution")
    plt.savefig('./cifar100newcifar100heatmap_homo.png')
    # plt.show()


def _plot_sample_distribution(prob_dist):
    plt.figure(0)
    logging.info("list(prob_dist.keys()) = %s" % list(prob_dist.keys()))
    logging.info("prob_dist.values() = %s" % prob_dist.values())
    plt.bar(list(prob_dist.keys()), prob_dist.values(), color='g')
    plt.xlabel('Client number')
    plt.ylabel('local training dataset size')
    plt.title("Min = " + str(min(prob_dist.values())) + ", Max = " + str(max(prob_dist.values())))
    # plt.text(0, 1000, " Mean = " + str(statistics.mean(prob_dist.values())))
    # plt.text(0, 950, ' STD = ' + str(
    #     statistics.stdev(prob_dist.values())))
    plt.savefig('./cifar100newsample_distribution_homo.png')
    # plt.show()
    logging.info('Figure saved')

# generate the non-IID distribution for all methods
def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    # logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar100():
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform

def load_cifar100_data(datadir):
    train_transform, test_transform = _data_transforms_cifar100()

    cifar10_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=train_transform)
    cifar10_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)

        print(idxs)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 100
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero-fix":
        dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR100/net_dataidx_map.txt'
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR100/distribution.txt'
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_CIFAR100(datadir, train_bs, test_bs, dataidxs)


# for local devices
# dataset, datadir, train_bs, test_bs, train_dataidxs=None, valid_dataidxs=None, test_dataidxs=None
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_contrain=None, train_dataidxs=None, valid_dataidxs=None, test_dataidxs=None):
    return get_dataloader_test_CIFAR100(datadir, train_bs, test_bs, dataidxs_contrain, train_dataidxs, valid_dataidxs, test_dataidxs)


def get_dataloader_CIFAR100(datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = CIFAR100_truncated

    transform_train, transform_test = _data_transforms_cifar100()

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_test_CIFAR100(datadir, train_bs, test_bs,dataidxs_contrain=None, dataidxs_train=None, dataidxs_valid=None, dataidxs_test=None):
    dl_obj = CIFAR100_truncated

    transform_train, transform_test = _data_transforms_cifar100()
#To load test and valid from trai data
    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=True, transform=transform_test, download=True)
    valid_ds = dl_obj(datadir, dataidxs=dataidxs_valid, train=True, transform=transform_test, download=True)
    contrain_ds = dl_obj(datadir, dataidxs=dataidxs_contrain, train=True, transform=transform_train, download=True)
    # train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    # test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)
    # train_ds = dl_obj(datadir, dataidxs=dataidxs_train, transform=transform_train, download=True)
    # test_ds = dl_obj(datadir, dataidxs=dataidxs_test, transform=transform_test, download=True)
    # valid_ds = dl_obj(datadir, dataidxs=dataidxs_valid, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    contrain_dl = data.DataLoader(dataset=contrain_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    valid_dl = data.DataLoader(dataset=valid_ds, batch_size=test_bs, shuffle=False, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)
    return train_dl, valid_dl, test_dl, contrain_dl


def load_partition_data_distributed_cifar100(process_id, dataset, data_dir, partition_method, partition_alpha,
                                            client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(train_data_global)))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs)
        logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, len(train_data_local), len(test_data_local)))
        train_data_global = None
        test_data_global = None

    return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num

def partition_data_byclass(dataset, datadir, partition, n_nets, classes=5):
    logging.info("*********partition data  by classes***************")
    X_complete, y_complete, X_test, y_test = load_cifar100_data(datadir)
    logging.info(X_complete.shape)
    logging.info(y_complete.shape)

    K = 100
    N = y_complete.shape[0]
    logging.info("N = " + str(N))
    net_dataidx_map = {}

    ## Separate each class data
    idx_xlist = []
    for k in range(K):
        logging.info(np.where(y_complete == k)[0])
        idx_xlist.append(np.where(y_complete == k)[0])

    # N = 60000
    # n_nets = 20
    # classes = 5
    # K = 10
    images_perclass = int(N / (n_nets * classes))  # 600
    logging.info(" images per class " + str(images_perclass))

    logging.info("idx_xlist = %s" % str(idx_xlist))
    logging.info("idx_xlist[0] = %s" % str(idx_xlist[0]))
    idx_batch = [[] for _ in range(n_nets)]

    class_assigned_fully_set = set()
    class_available_set = set([i for i in range(K)])
    n_nets = n_nets

    for j in range(n_nets):
        logging.info("class_assigned_fully_set = %s" % str(list(class_assigned_fully_set)))
        logging.info("class_available_set = %s" % str(class_available_set))
        # get the remaining class IDs that have not been assigned
        if len(class_available_set) < classes:
            logging.info("j = %d" % j)
            for l in class_available_set:
                logging.info("len of class %d = %d" % (j, len(idx_xlist[l])))
                idx_batch[j] = idx_batch[j] + idx_xlist[l].tolist()  # look how to partition it.
        else:
            classes_picked = random.sample(list(class_available_set), classes)
            for l in classes_picked:
                idx_batch[j] = idx_batch[j] + idx_xlist[l][0:images_perclass].tolist()  # look how to partition it.
                idx_xlist[l] = idx_xlist[l][images_perclass:]
                if len(idx_xlist[l]) == 0:
                    if l not in class_assigned_fully_set:
                        class_assigned_fully_set.add(l)
                        class_available_set.difference_update(list(class_assigned_fully_set))
                        logging.info("no samples left in class %d" % l)

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return X_complete, y_complete, net_dataidx_map, n_nets


def load_partition_data_cifar100(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    if partition_method == 'hetero':
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                                data_dir,
                                                                                                partition_method,
                                                                                                client_number,
                                                                                                partition_alpha)
    else:
        X_train, y_train, net_dataidx_map, n_nets = partition_data_byclass(dataset, data_dir,
                                                                                 partition_method,
                                                                                 client_number, 100)                                                                                           
    class_num = len(np.unique(y_train))
    # logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    # logging.info("train_dl_global number = " + str(len(train_data_global)))
    # logging.info("test_dl_global number = " + str(len(train_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    valid_data_local_dict = dict()
    contrain_data_local_dict = dict()
    prob_dist = {}
    train_data_number = 0

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        random.shuffle(dataidxs)

        # # Make three splits
        # train_len = int(0.50 * local_data_num) #50 #80
        # valid_len = int(0.30 * local_data_num) #30 #20
        # test_len = int(0.20 * local_data_num)  #20 #20
        #
        # data_local_num_dict[client_idx] = train_len
        #
        # # for plot
        # prob_dist[client_idx] = local_data_num
        # train_data_number += train_len
        #
        # # train
        # train_dataidxs = dataidxs[:train_len]
        # valid_dataidxs = dataidxs[train_len:valid_len+train_len]
        # test_dataidxs = dataidxs[valid_len+train_len:]


        # Make three splits
        # train_len = int(0.50 * local_data_num)  # 50 #80
        # valid_len = int(0.30 * local_data_num)  # 30 #20
        # test_len = int(0.20 * local_data_num)  # 20 #20
        # train_dataidxs = dataidxs[:train_len]
        # valid_dataidxs = dataidxs[train_len:train_len+valid_len]
        # test_dataidxs = dataidxs[train_len+valid_len:]


        train_len = int(0.60 * local_data_num) #50 #80
        valid_len = int(0.20 * local_data_num) #30 #20
        test_len = int(0.20 * local_data_num)  #20 #20
        

        data_local_num_dict[client_idx] = train_len

        # for plot
        prob_dist[client_idx] = local_data_num
        train_data_number += train_len

        # train
        train_dataidxs = dataidxs[:train_len]
        valid_dataidxs = dataidxs[train_len:train_len+valid_len]
        test_dataidxs = dataidxs[train_len+valid_len:]
        concat_train_dataidxs = dataidxs[:train_len+valid_len]
        train_data_local, valid_data_local, test_data_local, concat_data_local = get_dataloader_test(dataset, data_dir, batch_size, batch_size,
                                                                             concat_train_dataidxs, train_dataidxs, valid_dataidxs,
                                                                             test_dataidxs)

        train_data_local_dict[client_idx] = train_data_local
        valid_data_local_dict[client_idx] = valid_data_local
        test_data_local_dict[client_idx] = test_data_local
        contrain_data_local_dict[client_idx] = concat_data_local


        data_local_num_dict[client_idx] = local_data_num
        # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # # training batch size = 64; algorithms batch size = 32
        # train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
        #                                          dataidxs)
        # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        #     client_idx, len(train_data_local), len(test_data_local)))
        # train_data_local_dict[client_idx] = train_data_local
        # test_data_local_dict[client_idx] = test_data_local

    is_plot = False
    if is_plot:
        _plot_label_distribution(client_number, class_num, net_dataidx_map, y_train)
    logging.info(prob_dist)
    if is_plot:
        _plot_sample_distribution(prob_dist)
    # exit()
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, valid_data_local_dict, test_data_local_dict, class_num, contrain_data_local_dict
