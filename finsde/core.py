import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L

from sklearn.metrics import r2_score


class FinLightningModule(L.LightningModule):
    def __init__(
        self, 
        model, 
        loss=nn.MSELoss(),
        optimizer_partial=optim.Adam,
        lr_scheduler_partial=None,
        lr_scheduler_cfg=None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["model", "loss", "optimizer_partial", "lr_scheduler_partial"]
        )
        self.model = model
        self.loss = loss

        self.optimizer_partial = optimizer_partial
        self.lr_scheduler_partial = lr_scheduler_partial
        self.lr_scheduler_cfg = lr_scheduler_cfg if lr_scheduler_cfg else {}
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, torch.flatten(y, start_dim=1))
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, torch.flatten(y, start_dim=1))
        self.log("val_loss", loss)
        r2 = r2_score(
            torch.flatten(y, start_dim=0, end_dim=1).cpu().numpy(), 
            torch.flatten(y_hat.reshape(y.shape), start_dim=0, end_dim=1).cpu().numpy(),
        )
        self.log("val_r2", r2)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.reshape(y.shape)
        return y_hat
    
    def configure_optimizers(self):
        optimizer = self.optimizer_partial(
            self.parameters()
        )
        if self.lr_scheduler_partial is not None:
            scheduler = self.lr_scheduler_partial(
                optimizer,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    **self.lr_scheduler_cfg,
                },
            }
        else:
            return optimizer


class FinSDELightningModule(L.LightningModule):
    def __init__(
        self,
        model,
        loss=nn.MSELoss(),
        optimizer_partial=torch.optim.Adam,
        lr_scheduler_partial=None,
        lr_scheduler_cfg=None,
        forward_kwargs=dict(),
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["model", "optimizer_partial", "lr_scheduler_partial"]
        )
        self.model = model
        self.loss = loss
        self.forward_kwargs = forward_kwargs

        self.optimizer_partial = optimizer_partial
        self.lr_scheduler_partial = lr_scheduler_partial
        self.lr_scheduler_cfg = lr_scheduler_cfg if lr_scheduler_cfg else {}
    
    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        ts_in = torch.arange(0, x.shape[1], device=x.device) / x.shape[1]
        ts_out = torch.arange(0, x.shape[1] + y.shape[1], device=x.device) / x.shape[1]
        y_hat, likelihood, kl_div = self(x.permute(1, 0, 2), ts_in, ts_out, **self.forward_kwargs)
        loss = -likelihood + kl_div + self.loss(y_hat.permute(1, 0, 2), y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        ts_in = torch.arange(0, x.shape[1], device=x.device) / x.shape[1]
        ts_out = torch.arange(0, x.shape[1] + y.shape[1], device=x.device) / x.shape[1]
        y_hat, likelihood, kl_div = self(x.permute(1, 0, 2), ts_in, ts_out, **self.forward_kwargs)
        loss = -likelihood + kl_div + self.loss(y_hat.permute(1, 0, 2), y)
        self.log("val_loss", loss)
        y_hat = y_hat.permute(1, 0, 2)
        r2 = r2_score(
            torch.flatten(y, start_dim=0, end_dim=1).cpu().numpy(), 
            torch.flatten(y_hat.reshape(y.shape), start_dim=0, end_dim=1).cpu().numpy(),
        )
        self.log("val_r2", r2)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        ts_in = torch.arange(0, x.shape[1], device=x.device) / x.shape[1]
        ts_out = torch.arange(0, x.shape[1] + y.shape[1], device=x.device) / x.shape[1]
        y_hat, _, _ = self(x.permute(1, 0, 2), ts_in, ts_out, **self.forward_kwargs)
        y_hat = y_hat.permute(1, 0, 2)
        return y_hat
    
    def configure_optimizers(self):
        optimizer = self.optimizer_partial(
            self.parameters()
        )
        if self.lr_scheduler_partial is not None:
            scheduler = self.lr_scheduler_partial(
                optimizer,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    **self.lr_scheduler_cfg,
                },
            }
        else:
            return optimizer