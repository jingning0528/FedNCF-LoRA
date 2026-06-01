"""
FedNCF with server-side momentum on full item embedding matrix.

- Base model/client logic is reused from zoo.fedncf_base
- Only server aggregation for `embedding_item.weight` is changed
- Supported server_optimizer: "none", "ema", "heavyball"
"""

import copy
import logging
import torch

from dataloaders.BaseDataLoader import BaseDataLoaderFL
from framework.fed.server import ServerBase
from zoo.fedncf_base import model, Client, FedNCF_base


class Server(ServerBase):
    model: model

    def __init__(self, model, server_optimizer="ema", beta=0.9, eta_s=1.0):
        super().__init__(model)
        self.server_optimizer = str(server_optimizer).lower()  # none | ema | heavyball
        self.beta = float(beta)
        self.eta_s = float(eta_s)

        # Momentum buffer for full item embedding matrix
        self.v_item = torch.zeros_like(self.model.embedding_item.weight)

        logging.info(
            f"[FedNCF Full Momentum] method={self.server_optimizer}, beta={self.beta}, eta_s={self.eta_s}"
        )

    def aggregation(self, user_list, model_list, num_list, loss_list, cdp=None, ldp=None):
        self.model.eval()
        data_num = sum(num_list)
        base_model_dict = copy.deepcopy(self.model.state_dict())

        for name in base_model_dict.keys():
            if "embedding_user" in name:
                # Personalized user embedding rows
                for local_model, user in zip(model_list, user_list):
                    base_model_dict[name].data[user] = local_model[name].data[user]

            elif name == "embedding_item.weight":
                # FedAvg target for item embedding
                item_bar = sum([m[name] * n for m, n in zip(model_list, num_list)]) / data_num
                item_old = base_model_dict[name]
                delta = item_bar - item_old

                if self.server_optimizer == "none":
                    item_new = item_bar
                elif self.server_optimizer == "heavyball":
                    # v_t = beta * v_{t-1} + delta
                    # w_t = w_{t-1} + eta_s * v_t
                    self.v_item = self.beta * self.v_item.to(item_bar.device) + delta
                    item_new = item_old + self.eta_s * self.v_item
                else:
                    # EMA-style momentum:
                    # v_t = beta * v_{t-1} + (1-beta) * delta
                    # w_t = w_{t-1} + eta_s * v_t
                    self.v_item = self.beta * self.v_item.to(item_bar.device) + (1.0 - self.beta) * delta
                    item_new = item_old + self.eta_s * self.v_item

                base_model_dict[name] = item_new

                # Optional DP noise
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

            else:
                # Standard FedAvg for all other global params
                base_model_dict[name] = sum([m[name] * n for m, n in zip(model_list, num_list)]) / data_num

                # Optional DP noise
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
        logging.info("Clients average loss: {}".format(torch.mean(torch.tensor(loss_list))))


class FedNCF_Full_Momentum(FedNCF_base):
    """
    Reuse FedNCF_base training loop/model/client, but replace server aggregation
    with item-embedding momentum server optimizer.
    """

    def __init__(
        self,
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
        *args,
        **kwargs,
    ):
        super().__init__(
            dataload=dataload,
            clients_num_per_turn=clients_num_per_turn,
            local_epoch=local_epoch,
            train_turn=train_turn,
            user_num=user_num,
            item_num=item_num,
            embedding_dim=embedding_dim,
            hidden_activations=hidden_activations,
            hidden_units=hidden_units,
            output_dim=output_dim,
            latent_dim=latent_dim,
            device=device,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,
            task=task,
            *args,
            **kwargs,
        )

        # Keep current global model, replace only server logic
        self.server = Server(
            self.server.model,
            server_optimizer=kwargs.get("server_optimizer", "ema"),
            beta=kwargs.get("beta", 0.9),
            eta_s=kwargs.get("eta_s", 1.0),
        )