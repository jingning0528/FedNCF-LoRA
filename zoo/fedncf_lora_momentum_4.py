"""
Delta-B without momentum.
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
        self.delta_scale = float(kwargs.get("delta_scale", 1.0))  # constrain ΔB to be small
        self.lambda_delta = float(kwargs.get("lambda_delta", 1e-4))  # ← add this
        self.delta_B_lr           = float(kwargs.get("delta_B_lr", learning_rate))  # full lr

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

    def train_step_triple(self, users, pos, neg, global_model=None, update_delta_B=True):
        """
        Phase 2: B frozen, train A + ΔB + embedding_p + user_emb + MLP
        """
        self.train()
        self.embedding_item.linear.weight.requires_grad_(False)
        self.embedding_item.emb.weight.requires_grad_(True)
        self.delta_B.weight.requires_grad_(update_delta_B)
        self.embedding_p.weight.requires_grad_(True)   # ← changed: False → True

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

    def train_step_triple_c(self, users, pos, neg, global_model=None, update_delta_B=True):
        self.train()
        self.embedding_item.linear.weight.requires_grad_(False)
        self.embedding_item.emb.weight.requires_grad_(True)
        self.delta_B.weight.requires_grad_(update_delta_B)   # ← controlled
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
        delta_B_lr = getattr(self, "delta_B_lr", lr)  # fallback to lr if not set yet
        if optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam([
                {"params": self.embedding_user.parameters(),       "lr": lr},
                {"params": self.embedding_item.emb.parameters(),   "lr": lr},
                {"params": self.delta_B.parameters(),              "lr": delta_B_lr},  # configurable
                {"params": self.embedding_p.parameters(),          "lr": lr},
                {"params": self.mlp.parameters(),                  "lr": lr},
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

    def upload_model(self, update_delta_B=True):
        """
        Upload everything EXCEPT fixed B (embedding_item.linear).
        Skip ΔB if not being updated this round (saves communication).
        """
        full_state = self.model.state_dict()
        result = {}
        for k, v in full_state.items():
            if 'embedding_item.linear' in k:
                continue                              # B: frozen forever, never upload
            if 'delta_B' in k and not update_delta_B:
                continue                              # ΔB: skip if not updating this round
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
        self.models       = {}
        self.global_model = self.model.embedding_p.state_dict()
        self.delta_B_update_every = int(delta_B_update_every)
        self._turn = 0
        # no v_A — momentum removed

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
        update_delta_B = (self._turn % self.delta_B_update_every == 0)
        self._turn += 1

        for name in base_model_dict.keys():
            if 'embedding_item.linear' in name:
                continue

            elif 'embedding_user' in name:
                for m, user in zip(model_list, user_list):
                    base_model_dict[name].data[user] = m[name].data[user]

            elif 'embedding_item.emb' in name:
                # ===== A: standard FedAvg (no momentum) =====
                base_model_dict[name] = sum(
                    [m[name] * num for m, num in zip(model_list, num_list)]) / data_num
                # ============================================

            elif 'delta_B' in name:
                if update_delta_B:
                    delta_list = [(m[name], num) for m, num in zip(model_list, num_list) if name in m]
                    if len(delta_list) > 0:
                        base_model_dict[name] = sum(
                            [d * n for d, n in delta_list]) / sum([n for _, n in delta_list])
                        logging.info("ΔB updated at turn {} ({} clients)".format(
                            self._turn - 1, len(delta_list)))

            else:
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
        y_pred   = np.array(y_pred, np.float64)
        y_true   = np.array(y_true, np.float64)
        group_id = np.array(group_id) if len(group_id) > 0 else None
        val_logs = self.model.evaluate_metrics(y_true, y_pred, self.model.metrcis, group_id)
        logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v)
                                                for k, v in val_logs.items()))
        return val_logs


class FedNCF_Lora_Momentum_4:
    """
    No-momentum baseline: standard FedAvg for A + ΔB correction.
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
                delta_scale=float(kwargs.get("delta_scale", 0.3)),
                delta_B_lr=float(kwargs.get("delta_B_lr", learning_rate * 0.5)),
                lambda_delta=float(kwargs.get("lambda_delta", 1e-4)),
            )

        server_model = _make_model()
        server_model.reset_parameters()
        self.server = Server(
            server_model,
            delta_B_update_every=int(kwargs.get("delta_B_update_every", 10)),
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
        self.delta_B_update_every = int(kwargs.get("delta_B_update_every", 10))  # ← add this
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
            is_pretrain      = (turn < self.ae_warmup_turns)
            update_delta_B   = (not is_pretrain) and (turn % self.delta_B_update_every == 0)
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
                    update_delta_B=update_delta_B,   # ← pass flag
                )
                losses.append(loss)
                client_model.append(self.client.upload_model(update_delta_B))  # ← pass flag
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
