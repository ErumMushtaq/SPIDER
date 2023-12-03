import logging
import time

from fedml_api.distributed.SPIDER.message_define import MyMessage
from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message


class SPIDERClientManager(ClientManager):
    def __init__(self, args, comm, rank, size, trainer):
        super().__init__(args, comm, rank, size)

        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,self.__handle_msg_client_receive_config)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,self.__handle_msg_client_receive_model_from_server)

    def __handle_msg_client_receive_config(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        self.trainer.update_model(global_model_params)
        self.round_idx = 0
        self.__train()

    def __handle_msg_client_receive_model_from_server(self, msg_params):
        process_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        if process_id != 0:
            return
        self.trainer.update_dataset(int(client_index))
        self.trainer.update_model(model_params)
        self.round_idx += 1
        self.trainer.update_training_progress(self.round_idx)
        self.__train()
        if self.round_idx == self.num_rounds - 1:
            self.finish()

    def __train(self):
        weights, alphas, alpha_val, local_sample_num, local_acc, client_index, local_train_loss, reg_loss, flops, model_size = self.trainer.search()
        self.__send_msg_fedavg_send_model_to_server(weights,alphas, alpha_val, local_sample_num, local_acc, client_index, local_train_loss, reg_loss,flops, model_size)
        communication_finished_time = time.time()

    def __send_msg_fedavg_send_model_to_server(self, weights,alphas,alpha_val,local_sample_num, local_acc,  client_index, local_train_loss, reg_loss,flops, model_size):

        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.rank, 0)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_ARCH_PARAMS, alphas)
        message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_REG_LOSS, reg_loss)
        message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_TRAINING_LOSS, local_train_loss)
        message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_ACC, local_acc)
        message.add_params(MyMessage.MSG_ARG_KEY_FLOPS, flops)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_SIZE, model_size)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, client_index)
        message.add_params(MyMessage.MSG_ARG_KEY_ALPHA_VAL, alpha_val)
        self.send_message(message)