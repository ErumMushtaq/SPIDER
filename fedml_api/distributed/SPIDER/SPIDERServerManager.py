import logging

import torch

from fedml_api.distributed.SPIDER.message_define import MyMessage
from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.server.server_manager import ServerManager


class SPIDERServerManager(ServerManager):
    def __init__(self, args, comm, rank, size, aggregator):
        super().__init__(args, comm, rank, size)

        self.round_num = args.comm_round
        self.round_idx = 0
        self.aggregator = aggregator

    def run(self):
        global_model = self.aggregator.get_model()
        global_model_params = global_model.state_dict()
        for process_id in range(1, self.size):
            self.__send_initial_config_to_client(process_id, global_model_params)
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.__handle_msg_server_receive_model_from_client_opt_send)

    def __send_initial_config_to_client(self, process_id, global_model_params):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), process_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        self.send_message(message)

    def __handle_msg_server_receive_model_from_client_opt_send(self, msg_params):
        process_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        arch_params = msg_params.get(MyMessage.MSG_ARG_KEY_ARCH_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        alpha_val = msg_params.get(MyMessage.MSG_ARG_KEY_ALPHA_VAL)
        local_train_loss = msg_params.get(MyMessage.MSG_ARG_KEY_LOCAL_TRAINING_LOSS)
        local_acc = msg_params.get(MyMessage.MSG_ARG_KEY_LOCAL_ACC)
        flops = msg_params.get(MyMessage.MSG_ARG_KEY_FLOPS)
        model_size = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_SIZE)
        reg_loss = msg_params.get(MyMessage.MSG_ARG_KEY_LOCAL_REG_LOSS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.aggregator.add_local_trained_result(process_id - 1, model_params, arch_params, alpha_val, local_sample_number,
                                                 local_acc, local_train_loss, reg_loss,flops, model_size)

        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            if self.args.stage == 'search':
                # if self.args.FL == 'True':
                global_model_params = self.aggregator.aggregate()
                if self.round_idx % self.args.frequency_of_the_test == 0:
                    self.aggregator.statistics(self.round_idx)
                    self.aggregator.record_training_statistics(self.round_idx)
                    self.aggregator.record_local_model_architectures()
                    self.aggregator.local_model_statistics()
                    # free all teh GPU memory cache
                    torch.cuda.empty_cache()
                    global_model = self.aggregator.get_model()
                    folder = self.args.personalized_model_path + "/global_model"
                    path = folder + '.pth'
                    torch.save(global_model.cpu().state_dict(), path)  # save the model

            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                self.finish()
                return

            for process_id in range(1, self.size):
                self.__send_model_to_client_message(process_id, global_model_params,client_indexes[process_id - 1])

    def __send_model_to_client_message(self, process_id, global_model_params, client_index):
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, 0, process_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)
