from mpi4py import MPI

from fedml_api.distributed.SPIDER.SPIDERAggregator import SPIDERAggregator
from fedml_api.distributed.SPIDER.SPIDERClientManager import SPIDERClientManager
from fedml_api.distributed.SPIDER.SPIDERServerManager import SPIDERServerManager
from fedml_api.distributed.SPIDER.SPIDER_Trainer import SPIDERTrainer

def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_SPIDER_distributed(process_id, worker_number, device, comm, global_model, local_model, train_data_num, train_data_global, test_data_global,
                 local_data_num, train_data_local, valid_data_local, test_data_local, contrain_data_local_dict, args):
    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, global_model, train_data_num, train_data_global,
                    test_data_global)
    else:
        init_client(args, device, comm, process_id, worker_number, global_model, local_model, train_data_num, local_data_num,
                    train_data_local, valid_data_local, test_data_local,contrain_data_local_dict,)


def init_server(args, device, comm, process_id, worker_number, model, train_data_num, train_data_global, test_data_global):
    # aggregator
    client_num = worker_number - 1
    aggregator = SPIDERAggregator(train_data_global, test_data_global, train_data_num, client_num, model, device, args)

    # start the distributed training
    server_manager = SPIDERServerManager(args, comm, process_id, worker_number, aggregator)
    server_manager.run()


def init_client(args, device, comm, process_id, worker_number, global_model, local_model, train_data_num, local_data_num, train_data_local, valid_data_local, test_data_local, contrain_data_local_dict,):
    # trainer
    client_ID = process_id - 1
    trainer = SPIDERTrainer(client_ID, train_data_local, valid_data_local, test_data_local, local_data_num, train_data_num, global_model, local_model, device,contrain_data_local_dict, args)

    client_manager = SPIDERClientManager(args, comm, process_id, worker_number, trainer)
    client_manager.run()
