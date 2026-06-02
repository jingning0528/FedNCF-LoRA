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
    def __init__(self, model, ):
        super().__init__(model)

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
                for model, user in zip(model_list, user_list):
                    base_model_dict[name].data[user] = model[name].data[user]
            else:
                base_model_dict[name] = sum([model[name] * num for model, num in zip(model_list, num_list)]) / data_num
                if cdp is not None and cdp > 0.:
                    base_model_dict[name] += torch.normal(0, cdp, size=base_model_dict[name].size()).to(self.model.device)
                elif ldp is not None and ldp > 0.:
                    noise_list = [torch.normal(0, ldp, size=base_model_dict[name].size()).to(self.model.device) for _ in range(len(user_list))]
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
        val_logs = self.model.evaluate_metrics(y_true, y_pred, self.model.metrcis, group_id)
        logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
        return val_logs

class FedNCF_Full_Analyze():
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
        self.server = Server(server_model)
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
        self.compressed = kwargs.get("compressed", False)
        self.cdp = kwargs.get("cdp", None)
        self.ldp = kwargs.get("ldp", None)

        # ---- item embedding tracking buffers ----
        self.prev_track_item = None
        self.init_track_item = None
        self.prev_track_delta_item = None
        self.track_eps = 1e-12

    # ---- tracking helpers (full item embedding) ----
    def _get_item_for_tracking(self):
        return self.server.model.embedding_item.weight.detach().clone()

    def _safe_fro_norm(self, x):
        return torch.norm(x, p="fro")

    def _safe_cosine(self, x, y):
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        denom = torch.norm(x_flat) * torch.norm(y_flat)
        if denom.item() < self.track_eps:
            return float("nan")
        return (torch.dot(x_flat, y_flat) / denom).item()

    def _log_item_convergence_metrics(self, turn):
        item_now = self._get_item_for_tracking()

        # initialize baseline
        if self.prev_track_item is None:
            self.prev_track_item = item_now
            self.init_track_item = item_now
            logging.info(
                f"[Item_Track] turn={turn} initialized tracking baseline. "
                f"item_F={self._safe_fro_norm(item_now).item():.8f}"
            )
            return

        delta_item = item_now - self.prev_track_item

        delta_item_F = self._safe_fro_norm(delta_item).item()
        prev_item_F = self._safe_fro_norm(self.prev_track_item).item()
        curr_item_F = self._safe_fro_norm(item_now).item()
        init_item_F = self._safe_fro_norm(self.init_track_item).item()

        # normalized matrix magnitude of update (relative change wrt previous magnitude)
        norm_delta_item = delta_item_F / (prev_item_F + self.track_eps)

        # normalized matrix magnitude of current matrix (wrt initial magnitude)
        norm_item_vs_init = curr_item_F / (init_item_F + self.track_eps)

        if self.prev_track_delta_item is None:
            cos_delta_prev = float("nan")
        else:
            cos_delta_prev = self._safe_cosine(delta_item, self.prev_track_delta_item)

        logging.info(
            f"[Item_Track] turn={turn} "
            f"delta_item_F={delta_item_F:.8f} "
            f"norm_delta_item={norm_delta_item:.8f} "
            f"cos_delta_prev={cos_delta_prev:.8f} "
            f"item_F={curr_item_F:.8f} "
            f"norm_item_vs_init={norm_item_vs_init:.8f}"
        )

        self.prev_track_item = item_now
        self.prev_track_delta_item = delta_item

    def fit(self,):
        self.server.count_parameters()

        # baseline before FL rounds
        self._log_item_convergence_metrics(turn=-1)

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

            # ---- track + evaluate every 10 rounds ----
            if (turn + 1) % 10 == 0:
                self._log_item_convergence_metrics(turn=turn)

                logging.info("********* Eval @ Turn {} *********".format(turn))
                eval_start = time.perf_counter()
                self.server.evaluate(self.dataload, range(self.user_num))
                logging.info(f"[Time] eval_time={time.perf_counter() - eval_start:.4f}s")

        logging.info("********* Test *********")
        results = self.server.evaluate(self.dataload, range(self.user_num))
        return results
