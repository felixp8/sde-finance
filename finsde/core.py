import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L


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
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = self.optimizer_partial(
            self.parameters()
        )
        if self.lr_scheduler_partial is not None:
            scheduler = self.lr_scheduler_partial(
                optimizer,
                **self.lr_scheduler_partial.kwargs
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