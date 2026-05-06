"""
Fixed-B LoRA for FedNCF.

Supports two modes controlled by config `compressed`:
- compressed=False: item_emb = embedding_p + B(A(item)); embedding_p is warmed up then frozen.
- compressed=True : item_emb = B(A(item)) only; embedding_p is not used, uploaded, aggregated, or evaluated.

In both modes, B = embedding_item.linear is fixed: it is initialized non-zero, never trained, never uploaded, and never aggregated.
"""

from collections import OrderedDict
import copy
import numpy as np
import torch.nn as nn
import torch
import logging
import time
from dataloaders.BaseDataLoader import *
from framework.fed.client import ClientBase
from framework.fed.server import ServerBase
from framework.modules.models import BaseModel, AE, PQ_VAE, RPQ_VAE
from framework.modules.layers import MLP_Block
from framework.utils import calculate_model_size
from thop import profile

class model(BaseModel):
    def __init__(self, 
                 user_num, 
                 item_num, 
                 embedding_dim, 
                 hidden_activations, 
                 hidden_units, 
                 latent_dim,
                 task,
                 device, 
                 embedding_regularizer, 
                 net_regularizer, 
                 learning_rate,
                 optimizer,
                 loss_fn,
                 metrics,
                 *args, **kwargs):
        super(__class__, self).__init__(device=device,
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  metrics=metrics)
        self.embedding_user = nn.Embedding(num_embeddings=user_num, embedding_dim=embedding_dim)
        self.embedding_item = nn.Sequential(OrderedDict([('emb', nn.Embedding(item_num, latent_dim)), 
                                                      ('linear', nn.Linear(latent_dim, embedding_dim, bias=False)),
                                                      ]))
        self.embedding_p = nn.Embedding(num_embeddings=item_num, embedding_dim=embedding_dim)
        self.mlp = MLP_Block(input_dim = embedding_dim * 2,
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             dropout_rates=.5,
                             )
        self.task = task
        self.fedop = optimizer
        self.output_activation= nn.Sigmoid()
        self.reset_parameters()
        self.__init_weight()
        self.compile(optimizer=optimizer, loss=loss_fn, lr=learning_rate)
        self.model_to_device()

    def __init_weight(self, ):
        nn.init.normal_(self.embedding_item.emb.weight, std=0.1)
        nn.init.normal_(self.embedding_item.linear.weight, std=0.01)  # fixed B must be non-zero
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_p.weight, std=0.1)

    def emb_item(self, item_id):
        return self.embedding_p(item_id) + self.embedding_item(item_id)

    def emb_item_c(self, item_id):
        return self.embedding_item(item_id)

    def forward(self, user_id, item_id):
        output = self.mlp(torch.cat([self.embedding_user(user_id),self.emb_item(item_id)], -1))

        if self.task != "triple":
            output = self.output_activation(output)
            if self.task == "regression":
                output = output * 4.0 + 1.0
            return output
        return output

    def forward_c(self, user_id, item_id):
        output = self.mlp(torch.cat([self.embedding_user(user_id),self.emb_item_c(item_id)], -1))

        if self.task != "triple":
            output = self.output_activation(output)
            if self.task == "regression":
                output = output * 4.0 + 1.0
            return output
        return output
    
    def forward_pre(self, user_id, item_id):
        output = self.mlp(torch.cat([self.embedding_user(user_id),self.embedding_p(item_id)], -1))

        if self.task != "triple":
            output = self.output_activation(output)
            if self.task == "regression":
                output = output * 4.0 + 1.0
            return output
        return output

    def train_step(self, users, items, label, global_model=None):
        self.train()
        self.optimizer.zero_grad()
        pred = self.forward(users, items).squeeze()
        loss = self.loss_fn(pred, label, reduction='mean') + self.add_regularization()
        loss.backward()
        if self.fedop == "fedprox":
            self.optimizer.step(global_model)
        else:
            self.optimizer.step()
        return loss
    
    def train_step_triple(self, users, pos, neg, global_model=None):
        """
        Phase 2, non-compressed fixed-B LoRA:
            item_emb = embedding_p + B(A(item))
        B is fixed, embedding_p is frozen after warmup, only A + user_emb + MLP train.
        """
        self.train()

        # Fixed-B LoRA settings
        self.embedding_item.linear.weight.requires_grad_(False)  # freeze B
        self.embedding_item.emb.weight.requires_grad_(True)      # train A
        self.embedding_p.weight.requires_grad_(False)            # freeze base embedding after warmup

        self.optimizer.zero_grad()
        pred_pos = self.forward(users, pos)
        pred_neg = self.forward(users, neg)

        if len(users) > 0:
            loss = self.loss_fn(pred_pos, pred_neg) + self.add_regularization_triple(
                self.embedding_user.weight[users[0]],
                self.emb_item(pos),
                self.emb_item(neg),
            )
        else:
            loss = self.loss_fn(pred_pos, pred_neg)

        loss.backward()

        if self.fedop == "fedprox":
            self.optimizer.step(global_model)
        else:
            self.optimizer.step()

        return loss

    def train_step_triple_c(self, users, pos, neg, global_model=None):
        """
        Phase 2, compressed fixed-B LoRA:
            item_emb = B(A(item))
        B is fixed, embedding_p is unused/frozen, only A + user_emb + MLP train.
        """
        self.train()

        # Fixed-B compressed settings
        self.embedding_item.linear.weight.requires_grad_(False)  # freeze B
        self.embedding_item.emb.weight.requires_grad_(True)      # train A
        self.embedding_p.weight.requires_grad_(False)            # not used in compressed mode

        self.optimizer.zero_grad()
        pred_pos = self.forward_c(users, pos)
        pred_neg = self.forward_c(users, neg)

        if len(users) > 0:
            loss = self.loss_fn(pred_pos, pred_neg) + self.add_regularization_triple(
                self.embedding_user.weight[users[0]],
                self.emb_item_c(pos),
                self.emb_item_c(neg),
            )
        else:
            loss = self.loss_fn(pred_pos, pred_neg)

        loss.backward()

        if self.fedop == "fedprox":
            self.optimizer.step(global_model)
        else:
            self.optimizer.step()

        return loss
    
    def train_step_triple_pre(self, users, pos, neg, global_model=None):
        """
        Warmup stage for non-compressed mode:
            item_emb = embedding_p only
        Train embedding_p + user_emb + MLP; freeze A and B.
        """
        self.train()

        self.embedding_item.linear.weight.requires_grad_(False)  # freeze B
        self.embedding_item.emb.weight.requires_grad_(False)     # freeze A during base warmup
        self.embedding_p.weight.requires_grad_(True)             # train base item embedding

        self.optimizer.zero_grad()
        pred_pos = self.forward_pre(users, pos)
        pred_neg = self.forward_pre(users, neg)

        if len(users) > 0:
            loss = self.loss_fn(pred_pos, pred_neg) + self.add_regularization_triple(
                self.embedding_user.weight[users[0]],
                self.embedding_p(pos),
                self.embedding_p(neg),
            )
        else:
            loss = self.loss_fn(pred_pos, pred_neg)

        loss.backward()

        if self.fedop == "fedprox":
            self.optimizer.step(global_model)
        else:
            self.optimizer.step()

        return loss

class Client(ClientBase):
    model: model
    def __init__(self, client_id, model, task, fedop):
        super().__init__(client_id, model)
        self.task = task.lower()
        self.fedop = fedop.lower()

    def load_model(self, model):
        super().load_model(model)
        self.model.to(self.model.device)
        if self.fedop == "fedprox":
            self.global_model = copy.deepcopy(self.model.state_dict())

    def upload_model(self, compressed=False):
        """
        Upload trainable/shared states.
        - Always skip fixed B: embedding_item.linear.
        - In compressed=True, also skip embedding_p because forward_c does not use it.
        """
        full_state = self.model.state_dict()
        upload_state = {}
        for k, v in full_state.items():
            if 'embedding_item.linear' in k:
                continue  # fixed B, never upload
            if compressed and 'embedding_p' in k:
                continue  # compressed mode does not use embedding_p
            upload_state[k] = v.clone()
        return upload_state

    def local_train(self, user, local_epoch, dataload, pre_train=False, compressed=False):
        self.model.train()
        if self.task == "triple":
            users, pos, neg = dataload.get_traindata(user)
            self.__local_data_num = users.size(0)
            for _ in range(local_epoch):
                if self.fedop == "fedprox":
                    if compressed:
                        loss = self.model.train_step_triple_c(users, pos, neg, self.global_model)
                    elif pre_train:
                        loss = self.model.train_step_triple_pre(users, pos, neg, self.global_model)
                    else:
                        loss = self.model.train_step_triple(users, pos, neg, self.global_model)

                else:
                    if compressed:
                        loss = self.model.train_step_triple_c(users, pos, neg)
                    elif pre_train:
                        loss = self.model.train_step_triple_pre(users, pos, neg)
                    else:
                        loss = self.model.train_step_triple(users, pos, neg)
        else:
            users, items, labels = dataload.get_traindata(user)
            self.__local_data_num = labels.size(0)
            for _ in range(local_epoch):
                if self.fedop == "fedprox":  # ✅ was self.fedprox (AttributeError)
                    loss = self.model.train_step(users, items, labels, self.global_model)
                else:
                    loss = self.model.train_step(users, items, labels)
        # logging.info("Client {} for user {}, train loss: {:.6f}".format(self.client_id, user, loss))
        return loss
    
    def local_data_num(self):
        return self.__local_data_num

class Server(ServerBase):
    model: model
    def __init__(self, model, compressed=False):
        super().__init__(model)
        self.models = {}
        self.global_model = self.model.embedding_p.state_dict()
        self.compressed = compressed
        # momentum removed

    def count_parameters(self):
        # flops, params = profile(self.model, inputs=(torch.tensor(0, dtype=torch.int64, device=self.model.device),
        #                                             torch.tensor(1, dtype=torch.int64, device=self.model.device)))
        # logging.info("FLOPs: {:.8f} MFLOPs".format(flops/ 1e6))
        # logging.info("Param: {:.8f} M".format(params/ 1e6))
        self.model.eval()
        base_model_dict = copy.deepcopy(self.model.state_dict())
        model_size = 0.
        for name in base_model_dict.keys():
            if "embedding_user" in name:
                continue
            else:
                _, param_size = calculate_model_size(base_model_dict[name])
                logging.info("Model {} size: {:.8f}MB".format(name, param_size))
                model_size += param_size
        self.model.load_weights(copy.deepcopy(base_model_dict))
        logging.info("Model all size: {:.8f}MB".format(model_size))
    
    def distribute_model(self, user):
        return super().distribute_model()
        
    def aggregation(self, user_list, model_list, num_list, loss_list, cdp=None, ldp=None):
        self.model.eval()
        data_num = sum(num_list)
        base_model_dict = copy.deepcopy(self.model.state_dict())

        for name in base_model_dict.keys():
            if 'embedding_item.linear' in name:
                # B is fixed/shared — never updated from clients
                continue

            elif self.compressed and 'embedding_p' in name:
                # compressed mode does not use embedding_p, and clients do not upload it
                continue

            elif 'embedding_user' in name:
                # per-user embedding: update only selected users
                for m, user in zip(model_list, user_list):
                    base_model_dict[name].data[user] = m[name].data[user]

            elif 'embedding_item.emb' in name:
                # A aggregation: plain FedAvg (no momentum)
                base_model_dict[name] = sum(
                    [m[name] * num for m, num in zip(model_list, num_list)]
                ) / data_num

            else:
                # mlp, embedding_p: standard FedAvg
                base_model_dict[name] = sum(
                    [m[name] * num for m, num in zip(model_list, num_list)]
                ) / data_num
                if cdp is not None and cdp > 0.:
                    base_model_dict[name] += torch.normal(
                        0, cdp, size=base_model_dict[name].size()
                    ).to(self.model.device)
                elif ldp is not None and ldp > 0.:
                    noise_list = [torch.normal(0, ldp, size=base_model_dict[name].size()).to(
                        self.model.device) for _ in range(len(user_list))]
                    base_model_dict[name] += torch.mean(torch.stack(noise_list), dim=0)

        self.model.load_weights(copy.deepcopy(base_model_dict))
        logging.info("Clients average loss: {}".format(
            torch.mean(torch.tensor([l.detach() if isinstance(l, torch.Tensor) else l for l in loss_list]))
        ))

    def get_client_model(self, user):
        if user in self.models:
            self.model.embedding_p.load_state_dict(self.models[user])
        else:
            self.model.embedding_p.load_state_dict(self.global_model)
        self.model.to(self.model.device)

    def evaluate(self, dataload, user_list):
        self.model.eval()
        y_pred = []
        y_true = []
        group_id = []
        for user in user_list:
            users, items, labels = dataload.get_testdata(user)
            if self.compressed:
                pred = self.model.forward_c(users, items)
            else:
                pred = self.model.forward(users, items)
            y_pred.extend(pred.data.cpu().numpy().reshape(-1))
            y_true.extend(labels.data.cpu().numpy().reshape(-1))
            group_id.extend(users.data.cpu().numpy().reshape(-1))
        y_pred = np.array(y_pred, np.float64)
        y_true = np.array(y_true, np.float64)
        group_id = np.array(group_id) if len(group_id) > 0 else None
        val_logs = self.model.evaluate_metrics(y_true, y_pred, self.model.metrcis, group_id)
        logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
        return val_logs

class FedNCF_Lora_FixedB:
    def __init__(self, 
                 dataload:BaseDataLoaderFL,
                 clients_num_per_turn, 
                 local_epoch, 
                 train_turn,
                 user_num,
                 item_num,
                 embedding_dim,
                 hidden_activations, 
                 hidden_units, 
                 output_dim, 
                 latent_dim,
                 device, 
                 embedding_regularizer, 
                 net_regularizer, 
                 learning_rate,
                 optimizer, 
                 loss_fn,
                 metrics,
                 task,
                 *args, **kwargs
                 ):
        server_model = model(
            user_num=user_num,
            item_num=item_num,
            embedding_dim=embedding_dim,
            hidden_activations=hidden_activations,
            hidden_units=hidden_units,
            output_dim=output_dim,
            latent_dim=latent_dim,
            task=task.lower(),
            device=device,
            embedding_regularizer=embedding_regularizer, 
            net_regularizer=net_regularizer, 
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,
        )
        server_model.reset_parameters()
        self.compressed = kwargs.get("compressed", False)
        self.server = Server(server_model, compressed=self.compressed)  # beta/eta_s removed
        self.client = Client(client_id=0, model=model(
            user_num=user_num,
            item_num=item_num,
            embedding_dim=embedding_dim,
            hidden_activations=hidden_activations,
            hidden_units=hidden_units,
            output_dim=output_dim,
            latent_dim=latent_dim,
            task=task.lower(),
            device=device,
            embedding_regularizer=embedding_regularizer, 
            net_regularizer=net_regularizer, 
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,
        ), task=task.lower(), fedop=optimizer.lower()) 
        self.g_model = AE(hidden_units = kwargs["g_hidden_units"],
                hidden_activations = kwargs["g_hidden_activations"],
                embedding_dim = kwargs["sen_embedding_dim"], 
                embedding_dim_latent = embedding_dim,
                device = device, 
                embedding_regularizer=0., 
                net_regularizer=1e-2, 
                learning_rate=1e-4,
                optimizer="adam",
                loss_fn = "mse_loss",)
        self.clients_num_per_turn = clients_num_per_turn
        self.local_epoch =  local_epoch
        self.train_turn = train_turn
        self.user_num = user_num
        self.task = task.lower()
        self.device = device
        self.dataload = dataload
        self.pre_epoch = kwargs["pre_epoch"]
        self.cdp = kwargs.get("cdp", None)
        self.ldp = kwargs.get("ldp", None)

    def fit(self,):
        self.server.count_parameters()
        logging.info("FixedB mode: compressed={}".format(self.compressed))
        if not self.compressed:
            item_feature = self.dataload.get_item_feature()
            for turn in range(self.pre_epoch):
                loss = self.g_model.train_step(item_feature)
            latent = self.g_model.get_latent(item_feature)
            self.server.model.embedding_p.weight.data = copy.deepcopy(latent.detach())
            self.server.global_model = self.server.model.embedding_p.state_dict()

        for turn in range(self.train_turn):
            logging.info("********* Train Turn {} *********".format(turn))

            round_start = time.perf_counter()
            local_train_time = 0.0

            select_users = self.server.select_clients(self.user_num, self.clients_num_per_turn)
            client_model = []
            client_local_data_num = []
            losses = []

            for user in select_users:
                self.client.load_client(user)
                self.client.load_model(self.server.distribute_model(user))

                t0 = time.perf_counter()
                loss = self.client.local_train(user, self.local_epoch, self.dataload, turn < 20, self.compressed)
                local_train_time += time.perf_counter() - t0

                losses.append(loss)
                client_model.append(self.client.upload_model(compressed=self.compressed))
                client_local_data_num.append(self.client.local_data_num())

            agg_start = time.perf_counter()
            self.server.aggregation(select_users, client_model, client_local_data_num, losses, self.cdp, self.ldp)
            agg_time = time.perf_counter() - agg_start

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            round_time = time.perf_counter() - round_start
            avg_client_train_time = local_train_time / len(select_users) if len(select_users) > 0 else 0.0
            logging.info(
                f"[Time] turn={turn} "
                f"local_train_time={local_train_time:.4f}s "
                f"avg_client_train_time={avg_client_train_time:.6f}s "
                f"aggregation_time={agg_time:.4f}s "
                f"round_time={round_time:.4f}s"
            )

            if (turn + 1) % 10 == 0:
                logging.info("********* Eval @ Turn {} *********".format(turn))
                self.server.evaluate(self.dataload, range(self.user_num))

        logging.info("********* Final Test *********")
        results = self.server.evaluate(self.dataload, range(self.user_num))
        return results
