"""
Fixed B LoRA version.
Minimum changes:
- Freeze B = embedding_item.linear
- Do not upload B
- Do not aggregate B
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
        super(__class__, self).__init__(
            device=device,
            embedding_regularizer=embedding_regularizer, 
            net_regularizer=net_regularizer,
            metrics=metrics
        )

        self.embedding_user = nn.Embedding(num_embeddings=user_num, embedding_dim=embedding_dim)

        self.embedding_item = nn.Sequential(OrderedDict([
            ('emb', nn.Embedding(item_num, latent_dim)), 
            ('linear', nn.Linear(latent_dim, embedding_dim, bias=False)),
        ]))

        self.embedding_p = nn.Embedding(num_embeddings=item_num, embedding_dim=embedding_dim)

        self.mlp = MLP_Block(
            input_dim=embedding_dim * 2,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activations=hidden_activations,
            dropout_rates=.5,
        )

        self.task = task
        self.fedop = optimizer
        self.output_activation = nn.Sigmoid()

        self.reset_parameters()
        self.__init_weight()
        self.compile(optimizer=optimizer, loss=loss_fn, lr=learning_rate)
        self.model_to_device()

    def __init_weight(self):
        nn.init.normal_(self.embedding_item.emb.weight, std=0.1)
        nn.init.zeros_(self.embedding_item.linear.weight)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_p.weight, std=0.1)

    def emb_item(self, item_id):
        return self.embedding_p(item_id) + self.embedding_item(item_id)

    def emb_item_c(self, item_id):
        return self.embedding_item(item_id)

    def forward(self, user_id, item_id):
        output = self.mlp(torch.cat([
            self.embedding_user(user_id),
            self.emb_item(item_id)
        ], -1))

        if self.task != "triple":
            output = self.output_activation(output)
            if self.task == "regression":
                output = output * 4.0 + 1.0
            return output

        return output

    def forward_c(self, user_id, item_id):
        output = self.mlp(torch.cat([
            self.embedding_user(user_id),
            self.emb_item_c(item_id)
        ], -1))

        if self.task != "triple":
            output = self.output_activation(output)
            if self.task == "regression":
                output = output * 4.0 + 1.0
            return output

        return output
    
    def forward_pre(self, user_id, item_id):
        output = self.mlp(torch.cat([
            self.embedding_user(user_id),
            self.embedding_p(item_id)
        ], -1))

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

        self.embedding_p.weight.requires_grad_(False)  # CHANGED: correctly freeze base embedding_p
        self.embedding_item.linear.weight.requires_grad_(False)  # CHANGED: fixed B, freeze LoRA B
        self.embedding_item.emb.weight.requires_grad_(True)  # CHANGED: keep LoRA A trainable

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
        self.train()

        self.embedding_p.weight.requires_grad_(False)  # CHANGED: correctly freeze base embedding_p
        self.embedding_item.linear.weight.requires_grad_(False)  # CHANGED: fixed B, freeze LoRA B
        self.embedding_item.emb.weight.requires_grad_(True)  # CHANGED: keep LoRA A trainable

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
        self.train()

        self.embedding_p.weight.requires_grad_(True)  # CHANGED: correctly train base embedding_p during warmup
        self.embedding_item.linear.weight.requires_grad_(False)  # CHANGED: keep B frozen during warmup
        self.embedding_item.emb.weight.requires_grad_(False)  # CHANGED: keep A frozen during warmup

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

    def upload_model(self):
        # CHANGED: do not upload fixed B = embedding_item.linear
        full_state = self.model.state_dict()
        upload_state = {
            k: v.clone()
            for k, v in full_state.items()
            if 'embedding_item.linear' not in k
        }
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
                if self.fedop == "fedprox":
                    loss = self.model.train_step(users, items, labels, self.global_model)  # CHANGED: fixed wrong pos/neg
                else:
                    loss = self.model.train_step(users, items, labels)

        return loss
    
    def local_data_num(self):
        return self.__local_data_num


class Server(ServerBase):
    model: model

    def __init__(self, model):
        super().__init__(model)
        self.models = {}
        self.global_model = self.model.embedding_p.state_dict()

    def count_parameters(self):
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

            if "embedding_item.linear" in name:
                # CHANGED: fixed B, do not aggregate B from clients
                continue

            elif "embedding_user" in name:
                for model, user in zip(model_list, user_list):
                    base_model_dict[name].data[user] = model[name].data[user]

            else:
                base_model_dict[name] = sum([
                    model[name] * num
                    for model, num in zip(model_list, num_list)
                ]) / data_num

                if cdp is not None and cdp > 0.:
                    base_model_dict[name] += torch.normal(
                        0,
                        cdp,
                        size=base_model_dict[name].size()
                    ).to(self.model.device)

                elif ldp is not None and ldp > 0.:
                    noise_list = [
                        torch.normal(
                            0,
                            ldp,
                            size=base_model_dict[name].size()
                        ).to(self.model.device)
                        for _ in range(len(user_list))
                    ]
                    base_model_dict[name] += torch.mean(torch.stack(noise_list), dim=0)

        self.model.load_weights(copy.deepcopy(base_model_dict))

        logging.info("Clients average loss: {}".format(torch.mean(torch.tensor(loss_list))))

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

            y_pred.extend(self.model.forward(users, items).data.cpu().numpy().reshape(-1))
            y_true.extend(labels.data.cpu().numpy().reshape(-1))
            group_id.extend(users.data.cpu().numpy().reshape(-1))

        y_pred = np.array(y_pred, np.float64)
        y_true = np.array(y_true, np.float64)
        group_id = np.array(group_id) if len(group_id) > 0 else None

        val_logs = self.model.evaluate_metrics(
            y_true,
            y_pred,
            self.model.metrcis,
            group_id
        )

        logging.info('[Metrics] ' + ' - '.join(
            '{}: {:.6f}'.format(k, v) for k, v in val_logs.items()
        ))

        return val_logs


class FedNCF_Lora_FixedB_Analyze:
    def __init__(self, 
                 dataload: BaseDataLoaderFL,
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
                 *args, **kwargs):

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

        self.server = Server(server_model)

        self.client = Client(
            client_id=0,
            model=model(
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
            ),
            task=task.lower(),
            fedop=optimizer.lower()
        )

        self.g_model = AE(
            hidden_units=kwargs["g_hidden_units"],
            hidden_activations=kwargs["g_hidden_activations"],
            embedding_dim=kwargs["sen_embedding_dim"], 
            embedding_dim_latent=embedding_dim,
            device=device, 
            embedding_regularizer=0., 
            net_regularizer=1e-2, 
            learning_rate=1e-4,
            optimizer="adam",
            loss_fn="mse_loss",
        )

        self.clients_num_per_turn = clients_num_per_turn
        self.local_epoch = local_epoch
        self.train_turn = train_turn
        self.user_num = user_num
        self.task = task.lower()
        self.device = device
        self.dataload = dataload
        self.pre_epoch = kwargs["pre_epoch"]
        self.compressed = kwargs.get("compressed", False)
        self.cdp = kwargs.get("cdp", None)
        self.ldp = kwargs.get("ldp", None)

        # ADDED: A/B tracking buffers
        self.prev_track_A = None
        self.prev_track_B = None
        self.prev_track_delta_A = None
        self.prev_track_delta_B = None
        self.track_eps = 1e-12

    # ADDED: helpers for A/B convergence
    def _get_A_B_for_tracking(self):
        A = self.server.model.embedding_item.emb.weight.detach().clone()
        B = self.server.model.embedding_item.linear.weight.detach().clone()
        return A, B

    def _safe_fro_norm(self, x):
        return torch.norm(x, p="fro")

    def _safe_cosine(self, x, y):
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        denom = torch.norm(x_flat) * torch.norm(y_flat)
        if denom.item() < self.track_eps:
            return float("nan")
        return (torch.dot(x_flat, y_flat) / denom).item()

    def _log_AB_convergence_metrics(self, turn):
        A_now, B_now = self._get_A_B_for_tracking()

        if self.prev_track_A is None or self.prev_track_B is None:
            self.prev_track_A = A_now
            self.prev_track_B = B_now
            logging.info(
                f"[AB_Track] turn={turn} initialized tracking baseline. "
                f"A_norm={self._safe_fro_norm(A_now).item():.8f}, "
                f"B_norm={self._safe_fro_norm(B_now).item():.8f}"
            )
            return

        delta_A = A_now - self.prev_track_A
        delta_B = B_now - self.prev_track_B

        delta_A_norm = self._safe_fro_norm(delta_A).item()
        delta_B_norm = self._safe_fro_norm(delta_B).item()

        A_prev_norm = self._safe_fro_norm(self.prev_track_A).item()
        B_prev_norm = self._safe_fro_norm(self.prev_track_B).item()

        norm_delta_A = delta_A_norm / (A_prev_norm + self.track_eps)
        norm_delta_B = delta_B_norm / (B_prev_norm + self.track_eps)

        cos_delta_A = float("nan") if self.prev_track_delta_A is None else self._safe_cosine(delta_A, self.prev_track_delta_A)
        cos_delta_B = float("nan") if self.prev_track_delta_B is None else self._safe_cosine(delta_B, self.prev_track_delta_B)

        # effective low-rank embedding change: E = A @ B^T
        E_now = torch.matmul(A_now, B_now.t())
        E_prev = torch.matmul(self.prev_track_A, self.prev_track_B.t())
        effective_embedding_delta_norm = self._safe_fro_norm(E_now - E_prev).item()

        logging.info(
            f"[AB_Track] turn={turn} "
            f"delta_A_F={delta_A_norm:.8f} "
            f"delta_B_F={delta_B_norm:.8f} "
            f"norm_delta_A={norm_delta_A:.8f} "
            f"norm_delta_B={norm_delta_B:.8f} "
            f"cos_delta_A_prev={cos_delta_A:.8f} "
            f"cos_delta_B_prev={cos_delta_B:.8f} "
            f"effective_embedding_delta_F={effective_embedding_delta_norm:.8f} "
            f"A_F={self._safe_fro_norm(A_now).item():.8f} "
            f"B_F={self._safe_fro_norm(B_now).item():.8f}"
        )

        self.prev_track_A = A_now
        self.prev_track_B = B_now
        self.prev_track_delta_A = delta_A
        self.prev_track_delta_B = delta_B

    def fit(self):
        fit_start = time.perf_counter()

        self.server.count_parameters()

        if not self.compressed:
            pre_start = time.perf_counter()
            item_feature = self.dataload.get_item_feature()

            for turn in range(self.pre_epoch):
                loss = self.g_model.train_step(item_feature)

            latent = self.g_model.get_latent(item_feature)
            self.server.model.embedding_p.weight.data = copy.deepcopy(latent.detach())
            self.server.global_model = self.server.model.embedding_p.state_dict()
            logging.info(f"[Time] pretrain_time={time.perf_counter() - pre_start:.4f}s")

        self._log_AB_convergence_metrics(turn=-1)

        for turn in range(self.train_turn):
            logging.info("********* Train Turn {} *********".format(turn))

            round_start = time.perf_counter()
            local_train_time = 0.0
            client_train_times = []  # ADDED

            select_users = self.server.select_clients(self.user_num, self.clients_num_per_turn)

            client_model = []
            client_local_data_num = []
            losses = []

            for user in select_users:
                self.client.load_client(user)
                self.client.load_model(self.server.distribute_model(user))

                t0 = time.perf_counter()
                loss = self.client.local_train(user, self.local_epoch, self.dataload, turn < 20, self.compressed)
                dt = time.perf_counter() - t0  # ADDED

                local_train_time += dt
                client_train_times.append(dt)  # ADDED

                losses.append(loss)
                client_model.append(self.client.upload_model())
                client_local_data_num.append(self.client.local_data_num())

            agg_start = time.perf_counter()
            self.server.aggregation(select_users, client_model, client_local_data_num, losses, self.cdp, self.ldp)
            agg_time = time.perf_counter() - agg_start

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            round_time = time.perf_counter() - round_start

            if len(client_train_times) > 0:
                avg_client_train_time = local_train_time / len(client_train_times)
                max_client_train_time = max(client_train_times)
                min_client_train_time = min(client_train_times)
                median_client_train_time = float(np.median(client_train_times))
            else:
                avg_client_train_time = 0.0
                max_client_train_time = 0.0
                min_client_train_time = 0.0
                median_client_train_time = 0.0

            logging.info(
                f"[Time] turn={turn} "
                f"local_train_time={local_train_time:.4f}s "
                f"avg_client_train_time={avg_client_train_time:.6f}s "
                f"max_client_train_time={max_client_train_time:.6f}s "
                f"min_client_train_time={min_client_train_time:.6f}s "
                f"median_client_train_time={median_client_train_time:.6f}s "
                f"aggregation_time={agg_time:.4f}s "
                f"round_time={round_time:.4f}s"
            )

            if (turn + 1) % 10 == 0:
                self._log_AB_convergence_metrics(turn=turn)
                logging.info("********* Eval @ Turn {} *********".format(turn))
                logging.info(
                    f"[EvalRoundClientTime] turn={turn} "
                    f"avg={avg_client_train_time:.6f}s "
                    f"max={max_client_train_time:.6f}s "
                    f"min={min_client_train_time:.6f}s "
                    f"median={median_client_train_time:.6f}s"
                )

                eval_start = time.perf_counter()
                self.server.evaluate(self.dataload, range(self.user_num))
                logging.info(f"[Time] eval_time={time.perf_counter() - eval_start:.4f}s")

        logging.info("********* Final Test *********")
        final_eval_start = time.perf_counter()
        results = self.server.evaluate(self.dataload, range(self.user_num))
        logging.info(f"[Time] final_eval_time={time.perf_counter() - final_eval_start:.4f}s")
        logging.info(f"[Time] total_fit_time={time.perf_counter() - fit_start:.4f}s")
        return results