import numpy as np
import torch
import torch.utils.data as data
from sklearn.metrics import r2_score
from lightning.pytorch.callbacks import Callback


class R2Callback(Callback):
    def __init__(self, n_pred_steps=None, eval_dim=None, log_prefix="", split="val"):
        super().__init__()
        self.n_pred_steps = n_pred_steps
        self.log_prefix = log_prefix
        self.eval_dim = None if eval_dim is None else int(eval_dim)
        assert split in ["val", "test"]
        self.split = split
    
    def on_validation_epoch_end(self, trainer, pl_module):
        eval_dataset = trainer.datamodule.val_dataset if self.split == "val" else trainer.datamodule.test_dataset
        val_r2 = []
        val_mse = []
        val_mae = []
        for dataset in eval_dataset.datasets:
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
        pl_module.log(f"{self.log_prefix}total_{self.split}_r2", val_r2)
        pl_module.log(f"{self.log_prefix}total_{self.split}_mse", val_mse)
        pl_module.log(f"{self.log_prefix}total_{self.split}_mae", val_mae)


class RecurrentPredR2Callback(Callback):
    def __init__(self, n_pred_steps=[1, 3, 10], eval_dim=None, log_prefix="", split="val"):
        super().__init__()
        self.n_pred_steps = n_pred_steps
        self.log_prefix = log_prefix
        self.eval_dim = None if eval_dim is None else int(eval_dim)
        assert split in ["val", "test"]
        self.split = split

    def on_validation_epoch_end(self, trainer, pl_module):
        eval_dataset = trainer.datamodule.val_dataset if self.split == "val" else trainer.datamodule.test_dataset
        val_r2 = [[] for _ in range(len(self.n_pred_steps))]
        val_mse = [[] for _ in range(len(self.n_pred_steps))]
        val_mae = [[] for _ in range(len(self.n_pred_steps))]
        for dataset in eval_dataset.datasets:
            # temporarily set to max horizon
            orig_pred_horizon = dataset.dataset.pred_horizon
            assert dataset.dataset.max_horizon >= max(self.n_pred_steps)
            dataset.dataset.pred_horizon = max(self.n_pred_steps)
            if len(dataset) < 100:
                continue
            dataloader = data.DataLoader(dataset, batch_size=trainer.datamodule.batch_size)
            y_true = [[] for _ in range(len(self.n_pred_steps))]
            y_pred = [[] for _ in range(len(self.n_pred_steps))]
            for batch in dataloader:
                x, y = batch
                assert y.shape[1] >= max(self.n_pred_steps)
                assert y.shape[-1] == x.shape[-1]
                y_hat = []
                x_curr = x
                for _ in range(max(self.n_pred_steps)):
                    y_curr = pl_module.predict_step(
                        batch=(x_curr.to(pl_module.device), y[:, :orig_pred_horizon, :].to(pl_module.device)),
                        batch_idx=None,
                    ).to(y.device)
                    y_curr = y_curr.reshape((y.shape[0], orig_pred_horizon, -1))[:, [0], :]
                    x_curr = torch.cat([x_curr[:, 1:, :], y_curr], dim=1)
                    y_hat.append(y_curr)
                y_hat = torch.cat(y_hat, dim=1)
                for i, n_steps in enumerate(self.n_pred_steps):
                    y_hat_curr = y_hat[:, :n_steps, :]
                    y_curr = y[:, :n_steps, :]
                    if self.eval_dim is None:
                        y_true[i].append(y_curr)
                        y_pred[i].append(y_hat_curr)
                    else:
                        y_true[i].append(y_curr[..., [self.eval_dim]])
                        y_pred[i].append(y_hat_curr[..., [self.eval_dim]])
            y_true = [torch.cat(arr, dim=0) for arr in y_true]
            y_pred = [torch.cat(arr, dim=0) for arr in y_pred]
            for i in range(len(self.n_pred_steps)):
                r2 = r2_score(
                    torch.flatten(y_true[i], start_dim=0, end_dim=1).cpu().numpy(),
                    torch.flatten(y_pred[i], start_dim=0, end_dim=1).cpu().numpy(),
                )
                val_r2[i].append(r2)
                mse = torch.nn.functional.mse_loss(
                    torch.flatten(y_true[i], start_dim=0, end_dim=1), 
                    torch.flatten(y_pred[i], start_dim=0, end_dim=1),
                ).cpu().item()
                val_mse[i].append(mse)
                mae = torch.nn.functional.l1_loss(
                    torch.flatten(y_true[i], start_dim=0, end_dim=1), 
                    torch.flatten(y_pred[i], start_dim=0, end_dim=1),
                ).cpu().item()
                val_mae[i].append(mae)
            # reset pred_horizon
            dataset.dataset.pred_horizon = orig_pred_horizon
        for i, n_steps in enumerate(self.n_pred_steps):
            val_r2_mean = np.mean(val_r2[i]).item()
            val_mse_mean = np.mean(val_mse[i]).item()
            val_mae_mean = np.mean(val_mae[i]).item()
            pl_module.log(f"{self.log_prefix}total_{n_steps}step_{self.split}_r2", val_r2_mean)
            pl_module.log(f"{self.log_prefix}total_{n_steps}step_{self.split}_mse", val_mse_mean)
            pl_module.log(f"{self.log_prefix}total_{n_steps}step_{self.split}_mae", val_mae_mean)