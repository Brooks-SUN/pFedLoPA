import copy

import numpy as np
import time
import torch
# from flcore.clients.clientbase import Client
from flcore.clients.clientfedlora import clientLoRA
from utils.load_model import LoRALayer, get_lora_params, calculate_communication_size
from typing import List, Dict


class clientLoPA(clientLoRA):
    def __init__(self, args, lora_modules, id, train_samples, test_samples, **kwargs):
        super().__init__(args, lora_modules, id, train_samples, test_samples, **kwargs)
        self.pruning_ratio = args.pruning_ratio
        # self.lora_modules = lora_modules
        # self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0, 'comm_cost': 0.0}
        self.importance_accum = None

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        self.reset_importance()

        for name, param in self.model.named_parameters():
            if 'lora_A' not in name and 'lora_B' not in name:
                param.requires_grad = False

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.accumulate_importance()
                self.optimizer.step()

        self.apply_local_pruning()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def accumulate_importance(self):

        for name, module in self.model.named_modules():
            if isinstance(module, LoRALayer) and name in self.importance_accum:
                if module.lora_A.grad is not None:
                    self.importance_accum[name]['A'] += module.lora_A.grad.abs().detach()
                if module.lora_B.grad is not None:
                    self.importance_accum[name]['B'] += module.lora_B.grad.abs().detach()
                self.importance_accum[name]['count'] += 1

    def reset_importance(self):

        if self.importance_accum is None:
            self.importance_accum = {}
            device = next(self.model.parameters()).device

            for name, module in self.model.named_modules():
                if isinstance(module, LoRALayer):
                    self.importance_accum[name] = {
                        'A': torch.zeros_like(module.lora_A, device=device),
                        'B': torch.zeros_like(module.lora_B, device=device),
                        'count': 0
                    }

        for name in self.importance_accum:
            self.importance_accum[name]['A'].zero_()
            self.importance_accum[name]['B'].zero_()
            self.importance_accum[name]['count'] = 0

    def set_lora_params(self, lora_params: Dict):

        device = next(self.model.parameters()).device
        params_sizes, sparse_ratios = [], []
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALayer) and name in lora_params:
                client_params = copy.deepcopy(lora_params[name])
                un_mask_A = torch.ones_like(module.lora_A, device=device) - module.mask_A
                un_mask_B = torch.ones_like(module.lora_B, device=device) - module.mask_B
                client_params['lora_A'] = (module.lora_A * module.mask_A +
                                           lora_params[name]['lora_A'] * un_mask_A)
                client_params['lora_B'] = (module.lora_B * module.mask_B +
                                           lora_params[name]['lora_B'] * un_mask_B)
                module.set_lora_params(client_params)

                # calculate communication parameter size
                params_size, sparse_ratio = (
                    calculate_communication_size({name: {'mask_A': un_mask_A, 'mask_B': un_mask_B}}))
                params_sizes.append(params_size)
                sparse_ratios.append(sparse_ratio)
        return sum(params_sizes), sum(sparse_ratios)/len(sparse_ratios)

    def get_lora_params(self):
        client_lora_params = get_lora_params(self.model, self.lora_modules)
        for layer_name, params in client_lora_params.items():
            lora_A = params['lora_A']
            lora_B = params['lora_B']
            mask_A = params['mask_A']
            mask_B = params['mask_B']
            params['lora_A'] = lora_A * mask_A
            params['lora_B'] = lora_B * mask_B
        return client_lora_params

    def get_averaged_importance(self) -> Dict:

        importance_dict = {}
        for name, accum in self.importance_accum.items():
            if accum['count'] > 0:
                importance_dict[name] = {
                    'A': accum['A'] / accum['count'],
                    'B': accum['B'] / accum['count']
                }
            else:

                module = dict(self.model.named_modules())[name]
                importance_dict[name] = {
                    'A': module.lora_A.abs().detach(),
                    'B': module.lora_B.abs().detach()
                }
        return importance_dict

    def apply_local_pruning(self):

        importance_dict = self.get_averaged_importance()

        all_scores = []
        for name, scores in importance_dict.items():
            all_scores.append(scores['A'].flatten().cpu())
            all_scores.append(scores['B'].flatten().cpu())

        all_scores = torch.cat(all_scores)

        threshold = torch.quantile(all_scores, self.pruning_ratio)

        device = next(self.model.parameters()).device
        total_params = 0
        pruned_params = 0

        for name, module in self.model.named_modules():
            if isinstance(module, LoRALayer) and name in importance_dict:

                mask_A = (importance_dict[name]['A'].cpu() > threshold).float()
                mask_B = (importance_dict[name]['B'].cpu() > threshold).float()

                module.mask_A.copy_(mask_A.to(device))
                module.mask_B.copy_(mask_B.to(device))

                total_params += mask_A.numel() + mask_B.numel()
                pruned_params += (mask_A == 0).sum().item() + (mask_B == 0).sum().item()

