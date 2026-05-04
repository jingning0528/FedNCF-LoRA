"""
Delta-B without momentum.
Original idea: FedAvg for A + ΔB correction, with ΔB updated every few rounds to save communication and stabilize training.

Main fixes in this version:
1. Client and server now use the SAME update_delta_B flag from fit().
2. Server no longer recomputes update_delta_B using its own _turn counter.
3. ΔB schedule is based on Phase-2 rounds, not absolute training rounds.
   Example: ae_warmup_turns=20, delta_B_update_every=10
   ΔB updates at turn 20, 30, 40, ...
4. Added logging for ΔB norm and effective ΔB contribution.
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
    """
    LoRA with shared B + learnable correction ΔB:
        item_emb = embedding_p(i) + (B + delta_scale * ΔB) @ A(i)
                 = embedding_p(i) + B·A(i) + delta_scale * ΔB·A(i)

    B      : fixed/shared basis      (frozen forever)
    ΔB     : small global correction (updated only every delta_B_update_every rounds)
    A      : client low-rank coeff   (plain FedAvg, no momentum)
    """

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

        # A: item-side low-rank coefficients [item_num × latent_dim]
        self.embedding_item = nn.Sequential(OrderedDict([
            ('emb', nn.Embedding(item_num, latent_dim)),
            ('linear', nn.Linear(latent_dim, embedding_dim, bias=False)),  # B: fixed
        ]))

        # ΔB: small global correction [embedding_dim × latent_dim]
        self.delta_B = nn.Linear(latent_dim, embedding_dim, bias=False)
        self.delta_scale = float(kwargs.get("delta_scale", 0.3))
        self.lambda_delta = float(kwargs.get("lambda_delta", 1e-4))
        self.delta_B_lr = float(kwargs.get("delta_B_lr", learning_rate))

        self.embedding_user = nn.Embedding(num_embeddings=user_num, embedding_dim=embedding_dim)
        self.embedding_p = nn.Embedding(num_embeddings=item_num, embedding_dim=embedding_dim)

        self.mlp = MLP_Block(input_dim=embedding_dim * 2,
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             dropout_rates=.5)

        self.task = task
        self.fedop = optimizer
        self.output_activation = nn.Sigmoid()

        self.reset_parameters()
        self.__init_weight()
        self.compile(optimizer=optimizer, loss=loss_fn, lr=learning_rate)
        self.model_to_device()

    def __init_weight(self):
        nn.init.normal_(self.embedding_item.emb.weight, std=0.1)
        nn.init.normal_(self.embedding_item.linear.weight, std=0.01)  # fixed B, non-zero
        nn.init.zeros_(self.delta_B.weight)                           # ΔB starts from zero
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_p.weight, std=0.1)

    def _lora_item(self, item_id):
        """(B + delta_scale * ΔB) · A(item_id)"""
        a = self.embedding_item.emb(item_id)
        return self.embedding_item.linear(a) + self.delta_scale * self.delta_B(a)

    def emb_item(self, item_id):
        """Full item embedding: embedding_p + (B + delta_scale * ΔB)·A"""
        return self.embedding_p(item_id) + self._lora_item(item_id)

    def emb_item_c(self, item_id):
        """Compressed item embedding: (B + delta_scale * ΔB)·A only"""
        return self._lora_item(item_id)

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

    def train_step_triple(self, users, pos, neg, global_model=None, update_delta_B=True):
        """
        Phase 2: B frozen, train A + embedding_p + user_emb + MLP.
        ΔB is trainable only when update_delta_B=True.
        """
        self.train()
        self.embedding_item.linear.weight.requires_grad_(False)  # freeze B
        self.embedding_item.emb.weight.requires_grad_(True)      # train A
        self.delta_B.weight.requires_grad_(update_delta_B)       # train ΔB only on scheduled rounds
        self.embedding_p.weight.requires_grad_(True)

        self.optimizer.zero_grad()
        pred_pos = self.forward(users, pos)
        pred_neg = self.forward(users, neg)

        if len(users) > 0:
            loss = self.loss_fn(pred_pos, pred_neg) + self.add_regularization_triple(
                self.embedding_user.weight[users[0]], self.emb_item(pos), self.emb_item(neg))
        else:
            loss = self.loss_fn(pred_pos, pred_neg)

        if update_delta_B:
            loss = loss + self.lambda_delta * torch.mean(self.delta_B.weight ** 2)

        loss.backward()

        if self.fedop == "fedprox":
            self.optimizer.step(global_model)
        else:
            self.optimizer.step()

        return loss

    def train_step_triple_c(self, users, pos, neg, global_model=None, update_delta_B=True):
        self.train()
        self.embedding_item.linear.weight.requires_grad_(False)  # freeze B
        self.embedding_item.emb.weight.requires_grad_(True)      # train A
        self.delta_B.weight.requires_grad_(update_delta_B)       # train ΔB only on scheduled rounds
        self.embedding_p.weight.requires_grad_(False)

        self.optimizer.zero_grad()
        pred_pos = self.forward_c(users, pos)
        pred_neg = self.forward_c(users, neg)

        if len(users) > 0:
            loss = self.loss_fn(pred_pos, pred_neg) + self.add_regularization_triple(
                self.embedding_user.weight[users[0]], self.emb_item_c(pos), self.emb_item_c(neg))
        else:
            loss = self.loss_fn(pred_pos, pred_neg)

        if update_delta_B:
            loss = loss + self.lambda_delta * torch.mean(self.delta_B.weight ** 2)

        loss.backward()

        if self.fedop == "fedprox":
            self.optimizer.step(global_model)
        else:
            self.optimizer.step()

        return loss

    def train_step_triple_pre(self, users, pos, neg, global_model=None):
        """
        Phase 1: AE warmup — only embedding_p trains, LoRA A/B/ΔB frozen.
        """
        self.train()
        self.embedding_item.linear.weight.requires_grad_(False)
        self.embedding_item.emb.weight.requires_grad_(False)
        self.delta_B.weight.requires_grad_(False)
        self.embedding_p.weight.requires_grad_(True)

        self.optimizer.zero_grad()
        pred_pos = self.forward_pre(users, pos)
        pred_neg = self.forward_pre(users, neg)

        if len(users) > 0:
            loss = self.loss_fn(pred_pos, pred_neg) + self.add_regularization_triple(
                self.embedding_user.weight[users[0]], self.embedding_p(pos), self.embedding_p(neg))
        else:
            loss = self.loss_fn(pred_pos, pred_neg)

        loss.backward()

        if self.fedop == "fedprox":
            self.optimizer.step(global_model)
        else:
            self.optimizer.step()

        return loss

    def compile(self, optimizer, loss, lr):
        """Override compile to use separate LR for A and ΔB."""
        super().compile(optimizer=optimizer, loss=loss, lr=lr)
        delta_B_lr = getattr(self, "delta_B_lr", lr)

        if optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam([
                {"params": self.embedding_user.parameters(), "lr": lr},
                {"params": self.embedding_item.emb.parameters(), "lr": lr},
                {"params": self.delta_B.parameters(), "lr": delta_B_lr},
                {"params": self.embedding_p.parameters(), "lr": lr},
                {"params": self.mlp.parameters(), "lr": lr},
            ])
        elif optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD([
                {"params": self.embedding_user.parameters(), "lr": lr},
                {"params": self.embedding_item.emb.parameters(), "lr": lr},
                {"params": self.delta_B.parameters(), "lr": delta_B_lr},
                {"params": self.embedding_p.parameters(), "lr": lr},
                {"params": self.mlp.parameters(), "lr": lr},
            ])


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

    def upload_model(self, update_delta_B=True):
        """
        Upload everything EXCEPT fixed B.
        Skip ΔB if this is not a scheduled ΔB update round.
        """
        full_state = self.model.state_dict()
        result = {}

        for k, v in full_state.items():
            if 'embedding_item.linear' in k:
                continue  # fixed B, never upload
            if 'delta_B' in k and not update_delta_B:
                continue  # save communication on non-ΔB rounds
            result[k] = v.clone()

        return result

    def local_train(self, user, local_epoch, dataload, pre_train=False, compressed=False, update_delta_B=True):
        self.model.train()

        if self.task == "triple":
            users, pos, neg = dataload.get_traindata(user)
            self.__local_data_num = users.size(0)

            for _ in range(local_epoch):
                if self.fedop == "fedprox":
                    if compressed:
                        loss = self.model.train_step_triple_c(users, pos, neg, self.global_model, update_delta_B)
                    elif pre_train:
                        loss = self.model.train_step_triple_pre(users, pos, neg, self.global_model)
                    else:
                        loss = self.model.train_step_triple(users, pos, neg, self.global_model, update_delta_B)
                else:
                    if compressed:
                        loss = self.model.train_step_triple_c(users, pos, neg, update_delta_B=update_delta_B)
                    elif pre_train:
                        loss = self.model.train_step_triple_pre(users, pos, neg)
                    else:
                        loss = self.model.train_step_triple(users, pos, neg, update_delta_B=update_delta_B)
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

    def __init__(self, model, delta_B_update_every=10):
        super().__init__(model)
        self.models = {}
        self.global_model = self.model.embedding_p.state_dict()
        self.delta_B_update_every = int(delta_B_update_every)

    def count_parameters(self):
        self.model.eval()
        base_model_dict = copy.deepcopy(self.model.state_dict())
        model_size = 0.

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

    def aggregation(self, user_list, model_list, num_list, loss_list,
                    cdp=None, ldp=None, update_delta_B=False, turn=None):
        """
        Server aggregation.

        Important fix:
        - update_delta_B is passed from fit().
        - Server does NOT recompute update_delta_B with an internal counter.
        """
        self.model.eval()
        data_num = sum(num_list)
        base_model_dict = copy.deepcopy(self.model.state_dict())

        for name in base_model_dict.keys():
            if 'embedding_item.linear' in name:
                continue  # fixed B

            elif 'embedding_user' in name:
                for m, user in zip(model_list, user_list):
                    base_model_dict[name].data[user] = m[name].data[user]

            elif 'embedding_item.emb' in name:
                # A: standard FedAvg, no momentum
                base_model_dict[name] = sum(
                    [m[name] * num for m, num in zip(model_list, num_list)]
                ) / data_num

            elif 'delta_B' in name:
                if update_delta_B:
                    delta_list = [(m[name], num) for m, num in zip(model_list, num_list) if name in m]
                    if len(delta_list) > 0:
                        base_model_dict[name] = sum(
                            [d * n for d, n in delta_list]
                        ) / sum([n for _, n in delta_list])

                        delta_norm = base_model_dict[name].norm().item()
                        logging.info("ΔB updated at turn {} ({} clients), delta_B_norm={:.8f}".format(
                            turn, len(delta_list), delta_norm))
                    else:
                        logging.warning("update_delta_B=True at turn {}, but no client uploaded ΔB".format(turn))
                # else: keep server ΔB unchanged

            else:
                base_model_dict[name] = sum(
                    [m[name] * num for m, num in zip(model_list, num_list)]
                ) / data_num

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

        logging.info("Clients average loss: {}".format(
            torch.mean(torch.tensor([
                l.detach() if isinstance(l, torch.Tensor) else l
                for l in loss_list
            ]))
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
        logging.info('[Metrics] ' + ' - '.join(
            '{}: {:.6f}'.format(k, v) for k, v in val_logs.items()
        ))
        return val_logs


class FedNCF_Lora_DeltaB_V3:
    """
    No-momentum baseline: standard FedAvg for A + ΔB correction.
    item_emb = embedding_p(i) + (B + delta_scale * ΔB)·A(i)
    """

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
                delta_scale=float(kwargs.get("delta_scale", 0.3)),
                delta_B_lr=float(kwargs.get("delta_B_lr", learning_rate * 0.5)),
                lambda_delta=float(kwargs.get("lambda_delta", 1e-4)),
            )

        server_model = _make_model()
        server_model.reset_parameters()

        self.delta_B_update_every = int(kwargs.get("delta_B_update_every", 10))

        self.server = Server(
            server_model,
            delta_B_update_every=self.delta_B_update_every,
        )

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
        self.ae_warmup_turns = int(kwargs.get("ae_warmup_turns", 20))
        self.compressed = kwargs.get("compressed", False)
        self.cdp = kwargs.get("cdp", None)
        self.ldp = kwargs.get("ldp", None)

        logging.info("DeltaB config: delta_scale={}, delta_B_lr={}, delta_B_update_every={}, lambda_delta={}, ae_warmup_turns={}".format(
            float(kwargs.get("delta_scale", 0.3)),
            float(kwargs.get("delta_B_lr", learning_rate * 0.5)),
            self.delta_B_update_every,
            float(kwargs.get("lambda_delta", 1e-4)),
            self.ae_warmup_turns,
        ))

    def fit(self):
        self.server.count_parameters()
        logging.info("Phase schedule: Phase1(AE warmup)=turns 0-{}, Phase2(LoRA+ΔB)=turns {}+".format(
            self.ae_warmup_turns - 1, self.ae_warmup_turns
        ))

        # AE pre-training -> initialize embedding_p
        if not self.compressed:
            item_feature = self.dataload.get_item_feature()
            for t in range(self.pre_epoch):
                self.g_model.train_step(item_feature)
            latent = self.g_model.get_latent(item_feature)
            self.server.model.embedding_p.weight.data = copy.deepcopy(latent.detach())
            self.server.global_model = self.server.model.embedding_p.state_dict()

        for turn in range(self.train_turn):
            is_pretrain = (turn < self.ae_warmup_turns)

            # Important fix:
            # Use phase-2 relative turn, so ΔB update starts exactly when Phase 2 begins.
            if is_pretrain:
                update_delta_B = False
                phase2_turn = None
            else:
                phase2_turn = turn - self.ae_warmup_turns
                update_delta_B = (phase2_turn % self.delta_B_update_every == 0)

            if turn == self.ae_warmup_turns:
                logging.info("*** Phase 2 START: LoRA+ΔB — train A + scheduled ΔB, B frozen ***")

            phase_label = "phase1_AE" if is_pretrain else "phase2_LoRA+dB"
            logging.info("********* Train Turn {} [{}] *********".format(turn, phase_label))
            logging.info("Schedule: turn={}, phase2_turn={}, update_delta_B={}".format(
                turn, phase2_turn, update_delta_B
            ))

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
                    update_delta_B=update_delta_B,
                )
                local_train_time += time.perf_counter() - t0

                losses.append(loss)
                client_model.append(self.client.upload_model(update_delta_B=update_delta_B))
                client_local_data_num.append(self.client.local_data_num())

            agg_start = time.perf_counter()
            self.server.aggregation(
                select_users,
                client_model,
                client_local_data_num,
                losses,
                cdp=self.cdp,
                ldp=self.ldp,
                update_delta_B=update_delta_B,
                turn=turn,
            )
            agg_time = time.perf_counter() - agg_start
            round_time = time.perf_counter() - round_start

            # Debug ΔB after aggregation
            with torch.no_grad():
                delta_norm = self.server.model.delta_B.weight.norm().item()
                scaled_delta_norm = (self.server.model.delta_scale * self.server.model.delta_B.weight).norm().item()
                B_norm = self.server.model.embedding_item.linear.weight.norm().item()
            logging.info(
                "[DeltaB Debug] turn={} update_delta_B={} B_norm={:.8f} delta_B_norm={:.8f} scaled_delta_B_norm={:.8f}".format(
                    turn, update_delta_B, B_norm, delta_norm, scaled_delta_norm
                )
            )

            avg_client_train_time = (
                local_train_time / len(select_users) if len(select_users) > 0 else 0.0
            )
            logging.info(
                f"[Time] turn={turn} "
                f"phase={phase_label} "
                f"update_delta_B={update_delta_B} "
                f"local_train_time={local_train_time:.4f}s "
                f"avg_client_train_time={avg_client_train_time:.6f}s "
                f"aggregation_time={agg_time:.4f}s "
                f"round_time={round_time:.4f}s"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (turn + 1) % 10 == 0:
                logging.info("********* Eval @ Turn {} *********".format(turn))
                self.server.evaluate(self.dataload, range(self.user_num))

        logging.info("********* Final Test *********")
        results = self.server.evaluate(self.dataload, range(self.user_num))
        return results
