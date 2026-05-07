"""
Periodic B update, no Delta-B.

Goal:
    LoRA-like item embedding:
        item_emb = embedding_p(i) + B @ A(i)

    A = embedding_item.emb.weight        [item_num, latent_dim]
    B = embedding_item.linear.weight     [embedding_dim, latent_dim]

    Phase 1: AE warmup, turns 0 .. ae_warmup_turns-1
        - train embedding_p only
        - A and B frozen

    Phase 2: LoRA training, turns ae_warmup_turns+
        - A is trained/aggregated every round
        - B is trained/uploaded/aggregated periodically, controlled by B_update_every
        - For ae_warmup_turns=20 and B_update_every=15:
              B updates at turns 20, 35, 50, 65, ...
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
            metrics=metrics,
        )

        self.embedding_user = nn.Embedding(num_embeddings=user_num, embedding_dim=embedding_dim)

        # A: embedding_item.emb, B: embedding_item.linear
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
        nn.init.normal_(self.embedding_item.emb.weight, std=0.1)       # A
        nn.init.normal_(self.embedding_item.linear.weight, std=0.01)  # B: non-zero init
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_p.weight, std=0.1)

    def emb_item(self, item_id):
        return self.embedding_p(item_id) + self.embedding_item(item_id)

    def emb_item_c(self, item_id):
        return self.embedding_item(item_id)

    def forward(self, user_id, item_id):
        output = self.mlp(torch.cat([self.embedding_user(user_id), self.emb_item(item_id)], -1))
        if self.task != "triple":
            output = self.output_activation(output)
            if self.task == "regression":
                output = output * 4.0 + 1.0
        return output

    def forward_c(self, user_id, item_id):
        output = self.mlp(torch.cat([self.embedding_user(user_id), self.emb_item_c(item_id)], -1))
        if self.task != "triple":
            output = self.output_activation(output)
            if self.task == "regression":
                output = output * 4.0 + 1.0
        return output

    def forward_pre(self, user_id, item_id):
        output = self.mlp(torch.cat([self.embedding_user(user_id), self.embedding_p(item_id)], -1))
        if self.task != "triple":
            output = self.output_activation(output)
            if self.task == "regression":
                output = output * 4.0 + 1.0
        return output

    def set_lora_trainable(self, train_A=True, train_B=False):
        self.embedding_item.emb.weight.requires_grad_(train_A)
        self.embedding_item.linear.weight.requires_grad_(train_B)

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

    def train_step_triple(self, users, pos, neg, global_model=None, update_b=False):
        """
        Phase 2: train A every round; train B only when update_b=True.
        embedding_p, user_emb, and MLP remain trainable as in the original code.
        """
        self.train()
        self.set_lora_trainable(train_A=True, train_B=update_b)
        self.embedding_p.weight.requires_grad_(True)
        self.embedding_user.weight.requires_grad_(True)
        for p in self.mlp.parameters():
            p.requires_grad_(True)

        self.optimizer.zero_grad()
        pred_pos = self.forward(users, pos)
        pred_neg = self.forward(users, neg)
        if len(users) > 0:
            loss = self.loss_fn(pred_pos, pred_neg) + self.add_regularization_triple(
                self.embedding_user.weight[users[0]], self.emb_item(pos), self.emb_item(neg)
            )
        else:
            loss = self.loss_fn(pred_pos, pred_neg)
        loss.backward()
        if self.fedop == "fedprox":
            self.optimizer.step(global_model)
        else:
            self.optimizer.step()
        return loss

    def train_step_triple_c(self, users, pos, neg, global_model=None, update_b=False):
        """
        Compressed mode: train A every round; train B only when update_b=True.
        embedding_p is not used/frozen.
        """
        self.train()
        self.set_lora_trainable(train_A=True, train_B=update_b)
        self.embedding_p.weight.requires_grad_(False)
        self.embedding_user.weight.requires_grad_(True)
        for p in self.mlp.parameters():
            p.requires_grad_(True)

        self.optimizer.zero_grad()
        pred_pos = self.forward_c(users, pos)
        pred_neg = self.forward_c(users, neg)
        if len(users) > 0:
            loss = self.loss_fn(pred_pos, pred_neg) + self.add_regularization_triple(
                self.embedding_user.weight[users[0]], self.emb_item_c(pos), self.emb_item_c(neg)
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
        Phase 1: AE warmup — only embedding_p trains.
        """
        self.train()
        self.set_lora_trainable(train_A=False, train_B=False)
        self.embedding_p.weight.requires_grad_(True)
        self.embedding_user.weight.requires_grad_(True)
        for p in self.mlp.parameters():
            p.requires_grad_(True)

        self.optimizer.zero_grad()
        pred_pos = self.forward_pre(users, pos)
        pred_neg = self.forward_pre(users, neg)
        if len(users) > 0:
            loss = self.loss_fn(pred_pos, pred_neg) + self.add_regularization_triple(
                self.embedding_user.weight[users[0]], self.embedding_p(pos), self.embedding_p(neg)
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

    def upload_model(self, update_b=False):
        """
        Upload all trainable/shared parameters.
        Skip B when update_b=False, so B communication is periodic.
        """
        full_state = self.model.state_dict()
        result = {}
        for k, v in full_state.items():
            if 'embedding_item.linear' in k and not update_b:
                continue
            result[k] = v.clone()
        return result

    def local_train(self, user, local_epoch, dataload, pre_train=False, compressed=False, update_b=False):
        self.model.train()

        if self.task == "triple":
            users, pos, neg = dataload.get_traindata(user)
            self.__local_data_num = users.size(0)
            for _ in range(local_epoch):
                if self.fedop == "fedprox":
                    if compressed:
                        loss = self.model.train_step_triple_c(users, pos, neg, self.global_model, update_b=update_b)
                    elif pre_train:
                        loss = self.model.train_step_triple_pre(users, pos, neg, self.global_model)
                    else:
                        loss = self.model.train_step_triple(users, pos, neg, self.global_model, update_b=update_b)
                else:
                    if compressed:
                        loss = self.model.train_step_triple_c(users, pos, neg, update_b=update_b)
                    elif pre_train:
                        loss = self.model.train_step_triple_pre(users, pos, neg)
                    else:
                        loss = self.model.train_step_triple(users, pos, neg, update_b=update_b)
        else:
            users, items, labels = dataload.get_traindata(user)
            self.__local_data_num = labels.size(0)
            for _ in range(local_epoch):
                if self.fedop == "fedprox":
                    loss = self.model.train_step(users, items, labels, self.global_model)
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
        model_size = 0.0
        for name in base_model_dict.keys():
            if "embedding_user" in name:
                continue
            _, param_size = calculate_model_size(base_model_dict[name])
            logging.info("Model {} size: {:.8f}MB".format(name, param_size))
            model_size += param_size
        self.model.load_weights(copy.deepcopy(base_model_dict))
        logging.info("Model all size: {:.8f}MB".format(model_size))

    def distribute_model(self, user):
        return super().distribute_model()

    def aggregation(self, user_list, model_list, num_list, loss_list, update_b=False, cdp=None, ldp=None):
        """
        Aggregate client models.
        IMPORTANT: update_b is passed from fit(), so local training, upload,
        and server aggregation all use the same periodic B schedule.
        """
        self.model.eval()
        data_num = sum(num_list)
        base_model_dict = copy.deepcopy(self.model.state_dict())

        for name in base_model_dict.keys():
            if "embedding_user" in name:
                for m, user in zip(model_list, user_list):
                    base_model_dict[name].data[user] = m[name].data[user]

            elif "embedding_item.linear" in name:
                # B: aggregate only on periodic B update rounds.
                if update_b:
                    b_list = [(m[name], num) for m, num in zip(model_list, num_list) if name in m]
                    if len(b_list) > 0:
                        base_model_dict[name] = sum([b * n for b, n in b_list]) / sum([n for _, n in b_list])
                        logging.info("B updated this round ({} clients)".format(len(b_list)))
                    else:
                        logging.info("B update requested, but no clients uploaded B")
                else:
                    logging.info("B kept unchanged this round")

            else:
                # A, embedding_p, MLP: aggregate every round.
                base_model_dict[name] = sum([m[name] * num for m, num in zip(model_list, num_list)]) / data_num
                if cdp is not None and cdp > 0.0:
                    base_model_dict[name] += torch.normal(
                        0, cdp, size=base_model_dict[name].size()
                    ).to(self.model.device)
                elif ldp is not None and ldp > 0.0:
                    noise_list = [
                        torch.normal(0, ldp, size=base_model_dict[name].size()).to(self.model.device)
                        for _ in range(len(user_list))
                    ]
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
        y_pred, y_true, group_id = [], [], []
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


class FedNCF_Lora_PeriodicB:
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

        def _make_model():
            return model(
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

        server_model = _make_model()
        server_model.reset_parameters()
        self.server = Server(server_model)
        self.client = Client(
            client_id=0,
            model=_make_model(),
            task=task.lower(),
            fedop=optimizer.lower(),
        )

        self.g_model = AE(
            hidden_units=kwargs["g_hidden_units"],
            hidden_activations=kwargs["g_hidden_activations"],
            embedding_dim=kwargs["sen_embedding_dim"],
            embedding_dim_latent=embedding_dim,
            device=device,
            embedding_regularizer=0.0,
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

        # Configurable periodic-B schedule.
        self.ae_warmup_turns = int(kwargs.get("ae_warmup_turns", 20))
        self.B_update_every = int(kwargs.get("B_update_every", 10))

    def fit(self):
        fit_start = time.perf_counter()
        self.server.count_parameters()
        logging.info(
            "Phase schedule: Phase1(AE warmup)=turns 0-{}, Phase2(LoRA periodic B)=turns {}+".format(
                self.ae_warmup_turns - 1, self.ae_warmup_turns
            )
        )
        logging.info("B_update_every={} after warmup".format(self.B_update_every))

        # AE pre-training → initialise embedding_p.
        if not self.compressed:
            pre_start = time.perf_counter()
            item_feature = self.dataload.get_item_feature()
            for _ in range(self.pre_epoch):
                self.g_model.train_step(item_feature)
            latent = self.g_model.get_latent(item_feature)
            self.server.model.embedding_p.weight.data = copy.deepcopy(latent.detach())
            self.server.global_model = self.server.model.embedding_p.state_dict()
            logging.info(f"[Time] pretrain_time={time.perf_counter() - pre_start:.4f}s")

        for turn in range(self.train_turn):
            is_pretrain = (turn < self.ae_warmup_turns)
            phase_turn = turn - self.ae_warmup_turns
            update_b = (not is_pretrain) and (phase_turn % self.B_update_every == 0)

            if turn == self.ae_warmup_turns:
                logging.info("*** Phase 2 START: LoRA periodic B — train A every round, B periodically ***")

            phase_label = "phase1_AE" if is_pretrain else "phase2_LoRA_periodicB"
            logging.info("********* Train Turn {} [{}] *********".format(turn, phase_label))

            round_start = time.perf_counter()
            local_train_time = 0.0

            select_users = self.server.select_clients(self.user_num, self.clients_num_per_turn)
            client_model, client_local_data_num, losses = [], [], []

            for user in select_users:
                self.client.load_client(user)
                self.client.load_model(self.server.distribute_model(user))

                t0 = time.perf_counter()
                loss = self.client.local_train(
                    user,
                    self.local_epoch,
                    self.dataload,
                    pre_train=is_pretrain,
                    compressed=self.compressed,
                    update_b=update_b,
                )
                local_train_time += time.perf_counter() - t0

                losses.append(loss)
                client_model.append(self.client.upload_model(update_b=update_b))
                client_local_data_num.append(self.client.local_data_num())

            agg_start = time.perf_counter()
            self.server.aggregation(
                select_users,
                client_model,
                client_local_data_num,
                losses,
                update_b=update_b,
                cdp=self.cdp,
                ldp=self.ldp,
            )
            agg_time = time.perf_counter() - agg_start

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            round_time = time.perf_counter() - round_start
            avg_client_train_time = local_train_time / len(select_users) if len(select_users) > 0 else 0.0
            B_norm = self.server.model.embedding_item.linear.weight.norm().item()

            logging.info(
                f"[Time] turn={turn} "
                f"phase={phase_label} "
                f"phase_turn={phase_turn if not is_pretrain else -1} "
                f"update_b={update_b} "
                f"B_norm={B_norm:.6f} "
                f"local_train_time={local_train_time:.4f}s "
                f"avg_client_train_time={avg_client_train_time:.6f}s "
                f"aggregation_time={agg_time:.4f}s "
                f"round_time={round_time:.4f}s"
            )

            if (turn + 1) % 10 == 0:
                logging.info("********* Eval @ Turn {} *********".format(turn))
                eval_start = time.perf_counter()
                self.server.evaluate(self.dataload, range(self.user_num))
                logging.info(f"[Time] eval_time={time.perf_counter() - eval_start:.4f}s")

        logging.info("********* Final Test *********")
        final_eval_start = time.perf_counter()
        results = self.server.evaluate(self.dataload, range(self.user_num))
        logging.info(f"[Time] final_eval_time={time.perf_counter() - final_eval_start:.4f}s")
        logging.info(f"[Time] total_fit_time={time.perf_counter() - fit_start:.4f}s")
        return results
