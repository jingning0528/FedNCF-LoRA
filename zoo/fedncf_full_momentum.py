"""
Perifanis V, Efraimidis P S. Federated neural collaborative filtering
[J]. Knowledge-Based Systems, 2022, 242: 108441.

@article{DBLP:journals/kbs/PerifanisE22,
  author       = {Vasileios Perifanis and
                  Pavlos S. Efraimidis},
  title        = {Federated Neural Collaborative Filtering},
  journal      = {Knowl. Based Syst.},
  volume       = {242},
  pages        = {108441},
  year         = {2022},
  url          = {https://doi.org/10.1016/j.knosys.2022.108441},
  doi          = {10.1016/J.KNOSYS.2022.108441},
  timestamp    = {Fri, 22 Mar 2024 09:01:07 +0100},
  biburl       = {https://dblp.org/rec/journals/kbs/PerifanisE22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
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
        self.metrics = metrics  # <-- add this line (compatibility)
        self.embedding_user = nn.Embedding(num_embeddings=user_num, embedding_dim=embedding_dim)
        self.embedding_item = nn.Embedding(num_embeddings=item_num, embedding_dim=embedding_dim)
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
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def forward(self, user_id, item_id):
        output = self.mlp(torch.cat([self.embedding_user(user_id),self.embedding_item(item_id)], -1))

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
        self.train()
        self.optimizer.zero_grad()
        pred_pos = self.forward(users, pos)
        pred_neg = self.forward(users, neg)
        if len(users) > 0:
            loss = self.loss_fn(pred_pos, pred_neg, ) + self.add_regularization_triple(self.embedding_user.weight[users[0]], self.embedding_item(pos), self.embedding_item(neg),)
        else:
            loss = self.loss_fn(pred_pos, pred_neg, )
        loss.backward()
        if self.fedop == "fedprox":
            self.optimizer.step(global_model)
        else:
            self.optimizer.step()
        return loss

class Client(ClientBase):
    model:model
    def __init__(self, client_id, model, task, fedop):
        super().__init__(client_id, model)
        self.task = task.lower()
        self.fedop = fedop.lower()

    def load_model(self, model):
        super().load_model(model)
        self.model.to(self.model.device)
        if self.fedop == "fedprox":
            self.global_model = copy.deepcopy(self.model.state_dict())

    def local_train(self, user, local_epoch, dataload, pre_train=False, compressed=False):
        self.model.train()
        if self.task == "triple":
            users, pos, neg = dataload.get_traindata(user)
            self.__local_data_num = users.size(0)
            for _ in range(local_epoch):
                if self.fedop == "fedprox":
                    loss = self.model.train_step_triple(users, pos, neg, self.global_model)
                else:
                    loss = self.model.train_step_triple(users, pos, neg)
        else:
            users, items, labels = dataload.get_traindata(user)
            self.__local_data_num = labels.size(0)
            for _ in range(local_epoch):
                if self.fedop == "fedprox":
                    loss = self.model.train_step(users, pos, neg, self.global_model)
                else:
                    loss = self.model.train_step(users, items, labels)
        # logging.info("Client {} for user {}, train loss: {:.6f}".format(self.client_id, user, loss))
        return loss
    
    def local_data_num(self):
        return self.__local_data_num

class Server(ServerBase):
    model:model
    def __init__(self, model, server_optimizer="ema", beta=0.9, eta_s=1.0):
        super().__init__(model)
        self.server_optimizer = str(server_optimizer).lower()  # none | ema | heavyball
        self.beta = float(beta)
        self.eta_s = float(eta_s)

        # momentum buffer for item embedding matrix only
        self.v_item = torch.zeros_like(self.model.embedding_item.weight)

        logging.info(
            f"[FedNCF Full Momentum] method={self.server_optimizer}, beta={self.beta}, eta_s={self.eta_s}"
        )

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
            if "embedding_user" in name:
                for local_model, user in zip(model_list, user_list):
                    base_model_dict[name].data[user] = local_model[name].data[user]

            elif name == "embedding_item.weight":
                # FedAvg target
                item_bar = sum([m[name] * n for m, n in zip(model_list, num_list)]) / data_num
                item_old = base_model_dict[name]
                delta = item_bar - item_old

                if self.server_optimizer == "none":
                    item_new = item_bar
                elif self.server_optimizer == "heavyball":
                    # v_t = beta*v_{t-1} + delta
                    # w_t = w_{t-1} + eta_s*v_t
                    self.v_item = self.beta * self.v_item.to(item_bar.device) + delta
                    item_new = item_old + self.eta_s * self.v_item
                else:
                    # ema
                    # v_t = beta*v_{t-1} + (1-beta)*delta
                    # w_t = w_{t-1} + eta_s*v_t
                    self.v_item = self.beta * self.v_item.to(item_bar.device) + (1.0 - self.beta) * delta
                    item_new = item_old + self.eta_s * self.v_item

                base_model_dict[name] = item_new

                if cdp is not None and cdp > 0.:
                    base_model_dict[name] += torch.normal(
                        0, cdp, size=base_model_dict[name].size()
                    ).to(self.model.device)
                elif ldp is not None and ldp > 0.:
                    noise_list = [
                        torch.normal(0, ldp, size=base_model_dict[name].size()).to(self.model.device)
                        for _ in range(len(user_list))
                    ]
                    base_model_dict[name] += torch.mean(torch.stack(noise_list), dim=0)

            else:
                base_model_dict[name] = sum([m[name] * n for m, n in zip(model_list, num_list)]) / data_num
                if cdp is not None and cdp > 0.:
                    base_model_dict[name] += torch.normal(
                        0, cdp, size=base_model_dict[name].size()
                    ).to(self.model.device)
                elif ldp is not None and ldp > 0.:
                    noise_list = [
                        torch.normal(0, ldp, size=base_model_dict[name].size()).to(self.model.device)
                        for _ in range(len(user_list))
                    ]
                    base_model_dict[name] += torch.mean(torch.stack(noise_list), dim=0)

        self.model.load_weights(copy.deepcopy(base_model_dict))
        logging.info("Clients average loss: {}".format(torch.mean(torch.tensor(loss_list))))

    def evaluate(self, dataload, user_list):
        self.model.eval()
        y_pred = []
        y_true = []
        group_id = []
        for user in user_list:
            users, items, labels = dataload.get_testdata(user)
            y_pred.extend(self.model.forward(users, items).data.cpu().numpy().reshape(-1))
            y_true.extend(labels.data.cpu().numpy().reshape(-1))
            group_id.extend(users.data.cpu().numpy().reshape(-1))
        y_pred = np.array(y_pred, np.float64)
        y_true = np.array(y_true, np.float64)
        group_id = np.array(group_id) if len(group_id) > 0 else None

        eval_metrics = getattr(self.model, "metrics", getattr(self.model, "metrcis", None))
        val_logs = self.model.evaluate_metrics(y_true, y_pred, eval_metrics, group_id)

        logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
        return val_logs

class FedNCF_Full_Momentum:
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
        server_model =  model(
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

        self.server = Server(
            server_model,
            server_optimizer=kwargs.get("server_optimizer", "ema"),
            beta=kwargs.get("beta", 0.9),
            eta_s=kwargs.get("eta_s", 1.0),
        )

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

        # safe defaults to avoid KeyError
        kwargs.setdefault("g_hidden_units", [512, 256, 128])
        kwargs.setdefault("g_hidden_activations", hidden_activations)

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
        self.compressed = kwargs.get("compressed", False)
        self.cdp = kwargs.get("cdp", None)
        self.ldp = kwargs.get("ldp", None)

    def fit(self,):
        self.server.count_parameters()

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
                loss = self.client.local_train(user, self.local_epoch, self.dataload, False, self.compressed)
                local_train_time += time.perf_counter() - t0

                losses.append(loss)
                client_model.append(self.client.upload_model())
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

            # ---- evaluate every 10 rounds ----
            if (turn + 1) % 10 == 0:
                logging.info("********* Eval @ Turn {} *********".format(turn))
                eval_start = time.perf_counter()
                self.server.evaluate(self.dataload, range(self.user_num))
                logging.info(f"[Time] eval_time={time.perf_counter() - eval_start:.4f}s")

        logging.info("********* Test *********")
        results = self.server.evaluate(self.dataload, range(self.user_num))
        return results


