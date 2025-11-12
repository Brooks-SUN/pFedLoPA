
import wandb
import random
import time
from utils.data_utils import read_client_data
from flcore.clients.clientfedlopa import clientLoPA
from flcore.servers.serverbase import Server
from utils.load_model import calculate_communication_size, get_lora_params, LoRALayer


class FedLoPA(Server):
    def __init__(self, args, times, lora_modules):
        super().__init__(args, times)
        self.lora_modules = lora_modules
        self.uploaded_lora_models = []
        self.rs_train_acc = []
        self.rs_communication_cost = []

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientLoPA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_lora_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_lora_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_lora_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        # self.save_global_model()

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args,
                               self.lora_modules,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=train_slow,
                               send_slow=send_slow)
            self.clients.append(client)

    def send_lora_models(self):
        assert (len(self.clients) > 0)

        global_params = get_lora_params(self.global_model, self.lora_modules)
        for client in self.clients:
            start_time = time.time()

            params_size, sparse_ratio = client.set_lora_params(global_params)
            # params_size, sparse_ratio = calculate_communication_size(global_params)
            print(f'----Distribution client: {client.id} with sparsity: {sparse_ratio}----')

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            client.send_time_cost['comm_cost'] += params_size

    def receive_lora_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_lora_models = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            client_lora_params = client.get_lora_params()
            self.uploaded_lora_models.append(client_lora_params)
            params_size, sparse_ratio = calculate_communication_size(client_lora_params)
            print(f'----Upload client: {client.id} with sparsity: {sparse_ratio}----')
            client.send_time_cost['comm_cost'] += params_size

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_lora_parameters(self):
        assert (len(self.uploaded_lora_models) > 0)

        for name, module in self.global_model.named_modules():
            if isinstance(module, LoRALayer) and any(lora_name in name for lora_name in self.lora_modules):
                module.lora_A.data.zero_()
                module.lora_B.data.zero_()
                module.mask_A.data.zero_()
                module.mask_B.data.zero_()

        for w, client_lora_model in zip(self.uploaded_weights, self.uploaded_lora_models):
            for name, module in self.global_model.named_modules():
                if isinstance(module, LoRALayer) and any(lora_name in name for lora_name in self.lora_modules):
                    module.lora_A.data += client_lora_model[name]['lora_A'] * w
                    module.lora_B.data += client_lora_model[name]['lora_B'] * w
                    module.mask_A.data = module.mask_A.data.int() | client_lora_model[name]['mask_A'].int()
                    module.mask_B.data = module.mask_B.data.int() | client_lora_model[name]['mask_B'].int()

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        accs = []
        comm_cost = []
        for c in self.clients:
            acc, cl, cc, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)
            accs.append(acc * 1.0)
            comm_cost.append(cc/(1024 * 1024))

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses, accs, comm_cost

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        global_acc = self.global_test()
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_acc = sum(stats_train[3]) * 1.0 / sum(stats_train[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        comm_cost = sum(stats_train[4])

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        self.rs_test_auc.append(test_auc)
        self.rs_train_acc.append(train_acc)
        self.rs_global_acc.append(global_acc)
        self.rs_communication_cost.append(comm_cost)

        print("Global Accuracy: {:.4f}".format(global_acc))
        print("Averaged Train Accuracy: {:.4f}".format(train_acc))
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Communication cost: {:.4f}".format(comm_cost))

    def save_results(self):
        algo = f'{self.algorithm}_{self.args.model_str}_{self.dataset}_{self.args.pruning_ratio}'
        wandb.init(
            project="FedLoPA-pruning-ratio-experiments",
            name=algo,
        )

        if len(self.rs_test_acc):
            # logging in wandb
            for i in range(len(self.rs_test_acc)):
                wandb.log({'rs_global_acc': self.rs_global_acc[i]})
                wandb.log({'rs_test_acc': self.rs_test_acc[i]})
                wandb.log({'rs_test_auc': self.rs_test_auc[i]})
                wandb.log({'rs_train_acc': self.rs_train_acc[i]})
                wandb.log({'rs_train_loss': self.rs_train_loss[i]})
                wandb.log({'rs_comm_cost': self.rs_communication_cost[i]})
