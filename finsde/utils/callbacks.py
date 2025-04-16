import numpy as np
import torch
import torch.utils.data as data
from sklearn.metrics import r2_score
from lightning.pytorch.callbacks import Callback


class ValR2Callback(Callback):
    def __init__(self, n_pred_steps=None, eval_dim=None, log_prefix=""):
        super().__init__()
        self.n_pred_steps = n_pred_steps
        self.log_prefix = log_prefix
        self.eval_dim = None if eval_dim is None else int(eval_dim)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        val_dataset = trainer.datamodule.val_dataset
        val_r2 = []
        val_mse = []
        val_mae = []
        for dataset in val_dataset.datasets:
            if len(dataset) < 100:
                continue
            dataloader = data.DataLoader(dataset, batch_size=trainer.datamodule.batch_size)
            y_true = []
            y_pred = []
            for batch in dataloader:
                x, y = batch
                y_hat = pl_module.predict_step(
                    batch=(x.to(pl_module.device), y.to(pl_module.device)),
                    batch_idx=None,
                ).to(y.device)
                y_hat = y_hat.reshape(y.shape)
                if self.n_pred_steps is None:
                    y_true.append(y if self.eval_dim is None else y[..., [self.eval_dim]])
                    y_pred.append(y_hat if self.eval_dim is None else y_hat[..., [self.eval_dim]])
                else:
                    # Only keep the first n_pred_steps
                    y_true.append(y[:, :self.n_pred_steps, :] if self.eval_dim is None else y[:, :self.n_pred_steps, [self.eval_dim]])
                    y_pred.append(y_hat[:, :self.n_pred_steps, :] if self.eval_dim is None else y_hat[:, :self.n_pred_steps, [self.eval_dim]])
            y_true = torch.cat(y_true, dim=0)
            y_pred = torch.cat(y_pred, dim=0)
            r2 = r2_score(
                torch.flatten(y_true, start_dim=0, end_dim=1).cpu().numpy(),
                torch.flatten(y_pred, start_dim=0, end_dim=1).cpu().numpy(),
            )
            val_r2.append(r2)
            mse = torch.nn.functional.mse_loss(
                torch.flatten(y_true, start_dim=0, end_dim=1), 
                torch.flatten(y_pred, start_dim=0, end_dim=1),
            ).cpu().item()
            val_mse.append(mse)
            mae = torch.nn.functional.l1_loss(
                torch.flatten(y_true, start_dim=0, end_dim=1), 
                torch.flatten(y_pred, start_dim=0, end_dim=1),
            ).cpu().item()
            val_mae.append(mae)
        val_r2 = np.mean(val_r2).item()
        val_mse = np.mean(val_mse).item()
        val_mae = np.mean(val_mae).item()
        pl_module.log(f"{self.log_prefix}total_val_r2", val_r2)
        pl_module.log(f"{self.log_prefix}total_val_mse", val_mse)
        pl_module.log(f"{self.log_prefix}total_val_mae", val_mae)