"""
Perifanis V, Efraimidis P S. Federated neural collaborative filtering
[J]. Knowledge-Based Systems, 2022, 242: 108441.
"""

from collections import OrderedDict
import copy
import numpy as np
import torch.nn as nn
import torch
import logging
from dataloaders.BaseDataLoader import *
from framework.fed.client import ClientBase
from framework.fed.server import ServerBase
from framework.modules.models import BaseModel, AE, PQ_VAE, RPQ_VAE
from framework.modules.layers import MLP_Block
from framework.utils import calculate_model_size


class model(BaseModel):
    """
    LoRA with shared B + learnable correction ΔB:
        item_emb = embedding_p(i) + (B + ΔB) @ A(i)
                 = embedding_p(i) + B·A(i) + ΔB·A(i)
    
    B      : fixed/shared basis      (frozen forever)
    ΔB     : small global correction (lightly updated via FedAvg)
    A      : client low-rank coeff   (per-client, aggregated with momentum)
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
        # A: client-side low-rank coefficients  [item_num × latent_dim]
        self.embedding_item = nn.Sequential(OrderedDict([
            ('emb',    nn.Embedding(item_num, latent_dim)),
            ('linear', nn.Linear(latent_dim, embedding_dim, bias=False)),  # B: fixed
        ]))
        # ΔB: small global correction  [embedding_dim × latent_dim]
        self.delta_B = nn.Linear(latent_dim, embedding_dim, bias=False)
        self.delta_scale = float(kwargs.get("delta_scale", 0.1))  # constrain ΔB to be small
        self.lambda_delta = float(kwargs.get("lambda_delta", 1e-4))  # ← add this

        self.embedding_user = nn.Embedding(num_embeddings=user_num, embedding_dim=embedding_dim)
        self.embedding_p    = nn.Embedding(num_embeddings=item_num, embedding_dim=embedding_dim)
        self.mlp = MLP_Block(input_dim=embedding_dim * 2,
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             dropout_rates=.5)
        self.task   = task
        self.fedop  = optimizer
        self.output_activation = nn.Sigmoid()
        self.reset_parameters()
        self.__init_weight()
        self.compile(optimizer=optimizer, loss=loss_fn, lr=learning_rate)
        self.model_to_device()

    def __init_weight(self):
        nn.init.normal_(self.embedding_item.emb.weight, std=0.1)
        nn.init.normal_(self.embedding_item.linear.weight, std=0.01)  # fix 1: B nonzero, not zero
        nn.init.zeros_(self.delta_B.weight)                           # ΔB starts at zero (correction)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_p.weight, std=0.1)

    def _lora_item(self, item_id):
        """(B + delta_scale * ΔB) · A(item_id)"""
        a = self.embedding_item.emb(item_id)                                    # [N, latent_dim]
        return self.embedding_item.linear(a) + self.delta_scale * self.delta_B(a)  # fix 2: scale ΔB

    def emb_item(self, item_id):
        """Full item embedding: embedding_p + (B + ΔB)·A"""
        return self.embedding_p(item_id) + self._lora_item(item_id)

    def emb_item_c(self, item_id):
        """Compressed (no embedding_p): (B + ΔB)·A only"""
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

    def train_step_triple(self, users, pos, neg, global_model=None):
        """
        Phase 2: B frozen, train A + ΔB + user_emb + MLP
        """
        self.train()
        self.embedding_item.linear.weight.requires_grad_(False)
        self.embedding_item.emb.weight.requires_grad_(True)
        self.delta_B.weight.requires_grad_(True)
        self.embedding_p.weight.requires_grad_(False)

        self.optimizer.zero_grad()
        pred_pos = self.forward(users, pos)
        pred_neg = self.forward(users, neg)
        if len(users) > 0:
            loss = self.loss_fn(pred_pos, pred_neg) + self.add_regularization_triple(
                self.embedding_user.weight[users[0]], self.emb_item(pos), self.emb_item(neg))
        else:
            loss = self.loss_fn(pred_pos, pred_neg)
        # L2 regularization on ΔB to keep it small (mean-normalized, stable across dims)
        loss = loss + self.lambda_delta * torch.mean(self.delta_B.weight ** 2)
        loss.backward()
        if self.fedop == "fedprox":
            self.optimizer.step(global_model)
        else:
            self.optimizer.step()
        return loss

    def train_step_triple_c(self, users, pos, neg, global_model=None):
        self.train()
        self.embedding_item.linear.weight.requires_grad_(False)
        self.embedding_item.emb.weight.requires_grad_(True)
        self.delta_B.weight.requires_grad_(True)
        self.embedding_p.weight.requires_grad_(False)

        self.optimizer.zero_grad()
        pred_pos = self.forward_c(users, pos)
        pred_neg = self.forward_c(users, neg)
        if len(users) > 0:
            loss = self.loss_fn(pred_pos, pred_neg) + self.add_regularization_triple(
                self.embedding_user.weight[users[0]], self.emb_item_c(pos), self.emb_item_c(neg))
        else:
            loss = self.loss_fn(pred_pos, pred_neg)
        # L2 regularization on ΔB to keep it small (mean-normalized, stable across dims)
        loss = loss + self.lambda_delta * torch.mean(self.delta_B.weight ** 2)
        loss.backward()
        if self.fedop == "fedprox":
            self.optimizer.step(global_model)
        else:
            self.optimizer.step()
        return loss

    def train_step_triple_pre(self, users, pos, neg, global_model=None):
        """
        Phase 1: AE warmup — only embedding_p trains, everything else frozen
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
        # Let BaseModel handle loss_fn initialization
        super().compile(optimizer=optimizer, loss=loss, lr=lr)
        # Now override optimizer with per-param-group LR
        if optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam([
                {"params": self.embedding_user.parameters(),       "lr": lr},
                {"params": self.embedding_item.emb.parameters(),   "lr": lr},        # A: full lr
                {"params": self.delta_B.parameters(),              "lr": lr * 0.1},  # ΔB: 0.1x lr
                {"params": self.embedding_p.parameters(),          "lr": lr},
                {"params": self.mlp.parameters(),                  "lr": lr},
                # embedding_item.linear (B) intentionally excluded — frozen forever
            ])
        elif optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD([
                {"params": self.embedding_user.parameters(),       "lr": lr},
                {"params": self.embedding_item.emb.parameters(),   "lr": lr},
                {"params": self.delta_B.parameters(),              "lr": lr * 0.1},
                {"params": self.embedding_p.parameters(),          "lr": lr},
                {"params": self.mlp.parameters(),                  "lr": lr},
            ])
        # else: base class already raised or handled other optimizers


class Client(ClientBase):
    model: model

    def __init__(self, client_id, model, task, fedop):
        super().__init__(client_id, model)
        self.task  = task.lower()
        self.fedop = fedop.lower()

    def load_model(self, model):
        super().load_model(model)
        self.model.to(self.model.device)
        if self.fedop == "fedprox":
            self.global_model = copy.deepcopy(self.model.state_dict())

    def upload_model(self):
        """
        Upload everything EXCEPT fixed B (embedding_item.linear).
        - A (embedding_item.emb)  : per-client LoRA coeff
        - ΔB (delta_B)            : global correction
        - embedding_p             : needed for Phase 1 AE warmup aggregation
        - mlp, embedding_user     : standard FedAvg params
        """
        full_state = self.model.state_dict()
        return {k: v.clone() for k, v in full_state.items()
                if 'embedding_item.linear' not in k}   # exclude only fixed B

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
                    loss = self.model.train_step(users, items, labels, self.global_model)
                else:
                    loss = self.model.train_step(users, items, labels)
        return loss

    def local_data_num(self):
        return self.__local_data_num


class Server(ServerBase):
    model: model

    def __init__(self, model, beta=0.9, eta_s=1.0):
        super().__init__(model)
        self.models       = {}
        self.global_model = self.model.embedding_p.state_dict()
        self.beta         = beta
        self.eta_s        = eta_s
        # momentum velocity for A only
        self.v_A = torch.zeros_like(self.model.embedding_item.emb.weight)

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

    def aggregation(self, user_list, model_list, num_list, loss_list, cdp=None, ldp=None):
        self.model.eval()
        data_num = sum(num_list)
        base_model_dict = copy.deepcopy(self.model.state_dict())

        for name in base_model_dict.keys():

            if 'embedding_item.linear' in name:
                # B is permanently frozen — never updated
                continue

            elif 'embedding_user' in name:
                # per-user: update only selected users
                for m, user in zip(model_list, user_list):
                    base_model_dict[name].data[user] = m[name].data[user]

            elif 'embedding_item.emb' in name:
                # ===== A: FedAvg + Momentum =====
                A_bar   = sum([m[name] * num for m, num in zip(model_list, num_list)]) / data_num
                delta_A = A_bar - base_model_dict[name]
                self.v_A = self.beta * self.v_A.to(A_bar.device) + delta_A
                base_model_dict[name] = base_model_dict[name] + self.eta_s * self.v_A
                # ================================

            elif 'delta_B' in name:
                # ===== ΔB: standard FedAvg (light global correction) =====
                base_model_dict[name] = sum(
                    [m[name] * num for m, num in zip(model_list, num_list)]) / data_num
                # =========================================================

            else:
                # mlp, embedding_p: standard FedAvg
                base_model_dict[name] = sum(
                    [m[name] * num for m, num in zip(model_list, num_list)]) / data_num
                if cdp is not None and cdp > 0.:
                    base_model_dict[name] += torch.normal(
                        0, cdp, size=base_model_dict[name].size()).to(self.model.device)
                elif ldp is not None and ldp > 0.:
                    noise_list = [torch.normal(0, ldp, size=base_model_dict[name].size()).to(
                        self.model.device) for _ in range(len(user_list))]
                    base_model_dict[name] += torch.mean(torch.stack(noise_list), dim=0)

        self.model.load_weights(copy.deepcopy(base_model_dict))
        logging.info("Clients average loss: {}".format(
            torch.mean(torch.tensor([l.detach() if isinstance(l, torch.Tensor) else l
                                     for l in loss_list]))))

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
        y_pred    = np.array(y_pred, np.float64)
        y_true    = np.array(y_true, np.float64)
        group_id  = np.array(group_id) if len(group_id) > 0 else None
        val_logs  = self.model.evaluate_metrics(y_true, y_pred, self.model.metrcis, group_id)
        logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v)
                                                for k, v in val_logs.items()))
        return val_logs


class FedNCF_Lora_Momentum_3:
    """
    2-phase training:
      Phase 1 (turns 0–19)  : AE warmup  — embedding_p only
      Phase 2 (turns 20+)   : LoRA+ΔB   — train A + ΔB, B frozen forever

    item_emb = embedding_p(i) + (B + ΔB)·A(i)
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
                user_num=user_num, item_num=item_num,
                embedding_dim=embedding_dim,
                hidden_activations=hidden_activations,
                hidden_units=hidden_units,
                output_dim=output_dim,
                latent_dim=latent_dim,
                task=task.lower(), device=device,
                embedding_regularizer=embedding_regularizer,
                net_regularizer=net_regularizer,
                learning_rate=learning_rate,
                optimizer=optimizer,
                loss_fn=loss_fn, metrics=metrics,
                delta_scale=float(kwargs.get("delta_scale", 0.1)),  # fix 2: pass scale
            )

        server_model = _make_model()
        server_model.reset_parameters()
        self.server = Server(
            server_model,
            beta=float(kwargs.get("beta", 0.9)),
            eta_s=float(kwargs.get("eta_s", 1.0)),
        )
        self.client = Client(
            client_id=0, model=_make_model(),
            task=task.lower(), fedop=optimizer.lower(),
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
        self.local_epoch          = local_epoch
        self.train_turn           = train_turn
        self.user_num             = user_num
        self.task                 = task.lower()
        self.device               = device
        self.dataload             = dataload
        self.pre_epoch            = kwargs["pre_epoch"]
        self.ae_warmup_turns      = 20               # Phase 1: fixed
        self.compressed           = kwargs.get("compressed", False)
        self.cdp                  = kwargs.get("cdp", None)
        self.ldp                  = kwargs.get("ldp", None)

    def fit(self):
        self.server.count_parameters()
        logging.info("Phase schedule: Phase1(AE warmup)=turns 0-{}, Phase2(LoRA+ΔB)=turns {}+".format(
            self.ae_warmup_turns - 1, self.ae_warmup_turns))

        # AE pre-training → initialise embedding_p
        if not self.compressed:
            item_feature = self.dataload.get_item_feature()
            for t in range(self.pre_epoch):
                self.g_model.train_step(item_feature)
            latent = self.g_model.get_latent(item_feature)
            self.server.model.embedding_p.weight.data = copy.deepcopy(latent.detach())
            self.server.global_model = self.server.model.embedding_p.state_dict()

        for turn in range(self.train_turn):
            is_pretrain = (turn < self.ae_warmup_turns)   # Phase 1
            # Phase 2 starts at ae_warmup_turns
            if turn == self.ae_warmup_turns:
                logging.info("*** Phase 2 START: LoRA+ΔB — train A + ΔB, B frozen ***")

            phase_label = "phase1_AE" if is_pretrain else "phase2_LoRA+dB"
            logging.info("********* Train Turn {} [{}] *********".format(turn, phase_label))

            select_users = self.server.select_clients(self.user_num, self.clients_num_per_turn)
            client_model, client_local_data_num, losses = [], [], []

            for user in select_users:
                self.client.load_client(user)
                self.client.load_model(self.server.distribute_model(user))
                loss = self.client.local_train(
                    user, self.local_epoch, self.dataload,
                    pre_train=is_pretrain,
                    compressed=self.compressed,
                )
                losses.append(loss)
                client_model.append(self.client.upload_model())
                client_local_data_num.append(self.client.local_data_num())

            self.server.aggregation(
                select_users, client_model, client_local_data_num, losses,
                cdp=self.cdp, ldp=self.ldp,
            )
            torch.cuda.empty_cache()

            # ---- evaluate every 10 rounds ----
            if (turn + 1) % 10 == 0:
                logging.info("********* Eval @ Turn {} *********".format(turn))
                self.server.evaluate(self.dataload, range(self.user_num))

        logging.info("********* Final Test *********")
        results = self.server.evaluate(self.dataload, range(self.user_num))
        return results
