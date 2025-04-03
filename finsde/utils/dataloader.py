from pathlib import Path
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data
import torch.nn.functional as F
import lightning as L


class SingleStockDataset(data.Dataset):
    def __init__(
        self, 
        csv_file: str | Path, 
        use_returns=True, 
        feat_columns=["close", "volume"],
        pred_columns=["close"],
        pred_window=50,
        pred_horizon=3,
    ):
        self.use_returns = use_returns
        self.feat_columns = feat_columns
        self.pred_columns = pred_columns
        self.pred_window = pred_window
        self.pred_horizon = pred_horizon

        # Load and concatenate all CSV files into a single DataFrame
        self.all_columns = list(set(feat_columns + pred_columns))
        data_frame = pd.read_csv(csv_file)[self.all_columns]
        self.n_samples = len(data_frame - self.pred_window - self.pred_horizon + 1)

        # Preprocess the DataFrame
        feat, pred = self._preprocess_data(data_frame)
        self.feat = feat
        self.pred = pred
    
    def _preprocess_data(self, data_frame):
        # Convert to returns if required
        if self.use_returns and ("close" in self.all_columns):
            data_frame["close"] = data_frame["close"].pct_change()
        feat = torch.from_numpy(data_frame[self.feat_columns].to_numpy()).to(torch.get_default_dtype())
        pred = torch.from_numpy(data_frame[self.pred_columns].to_numpy()).to(torch.get_default_dtype())
        return feat, pred

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.feat[idx:idx + self.pred_window]
        y = torch.flatten(self.pred[idx + self.pred_window:idx + self.pred_window + self.pred_horizon])
        return x, y
    

class SingleStockDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path,
        csv_files: list[str],
        use_returns=True, 
        feat_columns=["close", "volume"],
        pred_columns=["close"],
        pred_window=50,
        pred_horizon=3,
        train_val_test_split=[0.6, 0.2, 0.2],
        batch_size=32,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.csv_files = csv_files
        self.use_returns = use_returns
        self.feat_columns = feat_columns
        self.pred_columns = pred_columns
        self.pred_window = pred_window
        self.pred_horizon = pred_horizon
        self.train_val_test_split = np.array(train_val_test_split) / np.sum(train_val_test_split)
        self.batch_size = batch_size

    def setup(self, stage: str):
        train_datasets = []
        val_datasets = []
        test_datasets = []
        for csv_file in self.csv_files:
            dataset = SingleStockDataset(
                csv_file=(self.data_dir / csv_file),
                use_returns=self.use_returns,
                feat_columns=self.feat_columns,
                pred_columns=self.pred_columns,
                pred_window=self.pred_window,
                pred_horizon=self.pred_horizon,
            )
            train_len = int(len(dataset.feat) * self.train_val_test_split[0])
            val_len = int(len(dataset.feat) * self.train_val_test_split[1])
            train_dataset = data.Subset(dataset, 
                range(train_len - self.pred_window - self.pred_horizon + 1))
            val_dataset = data.Subset(dataset,
                range(train_len, train_len + val_len - self.pred_window - self.pred_horizon + 1))
            test_dataset = data.Subset(dataset,
                range(train_len + val_len, len(dataset.feat) - self.pred_window - self.pred_horizon + 1))
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_datasets.append(test_dataset)
        self.train_dataset = data.ConcatDataset(train_datasets)
        self.val_dataset = data.ConcatDataset(val_datasets)
        self.test_dataset = data.ConcatDataset(test_datasets)
    
    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    