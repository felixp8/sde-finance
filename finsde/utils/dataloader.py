from pathlib import Path
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data
import lightning as L


class SingleStockDataset(data.Dataset):
    def __init__(
        self, 
        csv_file: str | Path, 
        feat_columns=["Close", "Volume"],
        pred_columns=["Close"],
        pred_window=50,
        pred_horizon=3,
        standardize=True,
        max_horizon=10,
    ):
        self.feat_columns = feat_columns
        self.pred_columns = pred_columns
        self.pred_window = pred_window
        self.pred_horizon = pred_horizon
        self.max_horizon = max_horizon
        self.standardize = standardize

        # Load and concatenate all CSV files into a single DataFrame
        self.all_columns = list(set(feat_columns + pred_columns) - set(["Returns"]))
        data_frame = pd.read_csv(csv_file)[self.all_columns]

        # Preprocess the DataFrame
        self.feat_scale = None
        self.feat_offset = None
        self.pred_scale = None
        self.pred_offset = None
        feat, pred = self._preprocess_data(data_frame)
        self.feat = feat
        self.pred = pred

        self.n_samples = (len(self.feat) - self.pred_window - self.max_horizon + 1)
    
    def _preprocess_data(self, data_frame):
        if "Returns" in self.feat_columns or "Returns" in self.pred_columns:
            data_frame["Returns"] = data_frame["Close"].pct_change()
        data_frame.dropna(inplace=True)
        feat = torch.from_numpy(data_frame[self.feat_columns].to_numpy()).to(torch.get_default_dtype())
        pred = torch.from_numpy(data_frame[self.pred_columns].to_numpy()).to(torch.get_default_dtype())
        if self.standardize:
            self.feat_offset = feat.min(dim=0, keepdim=True).values
            feat -= feat.min(dim=0, keepdim=True).values
            self.feat_scale = torch.clip(feat.max(dim=0, keepdim=True).values, min=1e-5)
            feat /= torch.clip(feat.max(dim=0, keepdim=True).values, min=1e-5)
            self.pred_offset = pred.min(dim=0, keepdim=True).values
            pred -= pred.min(dim=0, keepdim=True).values
            self.pred_scale = torch.clip(pred.max(dim=0, keepdim=True).values, min=1e-5)
            pred /= torch.clip(pred.max(dim=0, keepdim=True).values, min=1e-5)
        return feat, pred

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.feat[idx:idx + self.pred_window]
        y = self.pred[idx + self.pred_window:idx + self.pred_window + self.pred_horizon, :]
        return x, y
    

class SingleStockDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path,
        csv_files: list[str],
        feat_columns=["close", "volume"],
        pred_columns=["close"],
        pred_window=50,
        pred_horizon=3,
        max_horizon=10,
        standardize=True,
        train_val_test_split=[0.6, 0.2, 0.2],
        eval_csv_files: list[str] = [],
        batch_size=32,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.csv_files = csv_files
        self.feat_columns = feat_columns
        self.pred_columns = pred_columns
        self.pred_window = pred_window
        self.pred_horizon = pred_horizon
        self.max_horizon = max_horizon
        self.standardize = standardize
        self.train_val_test_split = np.array(train_val_test_split) / np.sum(train_val_test_split)
        self.batch_size = batch_size
        self.eval_csv_files = eval_csv_files

    def setup(self, stage: str):
        train_datasets = []
        val_datasets = []
        test_datasets = []
        use_test = self.train_val_test_split[2] > 1e-6
        for csv_file in self.csv_files:
            dataset = SingleStockDataset(
                csv_file=(self.data_dir / csv_file),
                feat_columns=self.feat_columns,
                pred_columns=self.pred_columns,
                pred_window=self.pred_window,
                pred_horizon=self.pred_horizon,
                max_horizon=self.max_horizon,
                standardize=self.standardize,
            )
            train_len = int(len(dataset.feat) * self.train_val_test_split[0])
            val_len = int(len(dataset.feat) * self.train_val_test_split[1])
            train_dataset = data.Subset(dataset, 
                range(train_len - self.pred_window - self.max_horizon + 1))
            train_datasets.append(train_dataset)
            if len(self.eval_csv_files) > 0:  # if only eval on subset then check
                if csv_file in self.eval_csv_files:
                    val_dataset = data.Subset(dataset,
                        range(train_len, train_len + val_len - self.pred_window - self.max_horizon + 1))
                    val_datasets.append(val_dataset)
                    if use_test:
                        test_dataset = data.Subset(dataset,
                            range(train_len + val_len, len(dataset.feat) - self.pred_window - self.max_horizon + 1))
                        test_datasets.append(test_dataset)
            else:  # otherwise add all val datasets
                val_dataset = data.Subset(dataset,
                    range(train_len, train_len + val_len - self.pred_window - self.max_horizon + 1))
                val_datasets.append(val_dataset)
                if use_test:
                    test_dataset = data.Subset(dataset,
                        range(train_len + val_len, len(dataset.feat) - self.pred_window - self.max_horizon + 1))
                    test_datasets.append(test_dataset)
        self.train_dataset = data.ConcatDataset(train_datasets)
        self.val_dataset = data.ConcatDataset(val_datasets)
        if use_test:
            self.test_dataset = data.ConcatDataset(test_datasets)
        else:
            self.test_dataset = None
    
    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class MultiStockDataset(data.Dataset):
    def __init__(
        self, 
        csv_file: str | Path, 
        pred_window=50,
        pred_horizon=3,
        standardize=True,
        max_horizon=10,
    ):
        self.pred_window = pred_window
        self.pred_horizon = pred_horizon
        self.max_horizon = max_horizon
        self.standardize = standardize

        # Load and concatenate all CSV files into a single DataFrame
        data_frame = pd.read_csv(csv_file)

        # Preprocess the DataFrame
        self.feat_scale = None
        self.feat_offset = None
        self.pred_scale = None
        self.pred_offset = None
        feat, pred = self._preprocess_data(data_frame)
        self.feat = feat
        self.pred = pred

        self.n_samples = (len(self.feat) - self.pred_window - self.max_horizon + 1)

    def _preprocess_data(self, data_frame):
        data_frame.dropna(inplace=True)
        data_frame.drop('Date', axis=1, inplace=True)
        feat = torch.from_numpy(data_frame.to_numpy()).to(torch.get_default_dtype())
        pred = torch.from_numpy(data_frame.to_numpy()).to(torch.get_default_dtype())
        if self.standardize:
            self.feat_offset = feat.min(dim=0, keepdim=True).values
            feat -= feat.min(dim=0, keepdim=True).values
            self.feat_scale = torch.clip(feat.max(dim=0, keepdim=True).values, min=1e-5)
            feat /= torch.clip(feat.max(dim=0, keepdim=True).values, min=1e-5)
            self.pred_offset = pred.min(dim=0, keepdim=True).values
            pred -= pred.min(dim=0, keepdim=True).values
            self.pred_scale = torch.clip(pred.max(dim=0, keepdim=True).values, min=1e-5)
            pred /= torch.clip(pred.max(dim=0, keepdim=True).values, min=1e-5)
        return feat, pred

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.feat[idx:idx + self.pred_window]
        y = self.pred[idx + self.pred_window:idx + self.pred_window + self.pred_horizon, :]
        return x, y


class MultiStockDataModule(L.LightningDataModule):
    def __init__(
        self,
        csv_file: str | Path,
        pred_window=50,
        pred_horizon=3,
        max_horizon=10,
        standardize=True,
        train_val_test_split=[0.6, 0.2, 0.2],
        batch_size=32,
        **kwargs,
    ):
        super().__init__()
        self.csv_file = csv_file
        self.pred_window = pred_window
        self.pred_horizon = pred_horizon
        self.max_horizon = max_horizon
        self.standardize = standardize
        self.train_val_test_split = np.array(train_val_test_split) / np.sum(train_val_test_split)
        self.batch_size = batch_size

    def setup(self, stage: str):
        use_test = self.train_val_test_split[2] > 1e-6
        dataset = MultiStockDataset(
            csv_file=(self.csv_file),
            pred_window=self.pred_window,
            pred_horizon=self.pred_horizon,
            standardize=self.standardize,
        )
        train_len = int(len(dataset.feat) * self.train_val_test_split[0])
        val_len = int(len(dataset.feat) * self.train_val_test_split[1])
        self.train_dataset = data.ConcatDataset([data.Subset(dataset, 
            range(train_len - self.pred_window - self.max_horizon + 1))])
        self.val_dataset = data.ConcatDataset([data.Subset(dataset,
            range(train_len, train_len + val_len - self.pred_window - self.max_horizon + 1))])
        if use_test:
            self.test_dataset = data.ConcatDataset([data.Subset(dataset,
                range(train_len + val_len, len(dataset.feat) - self.pred_window - self.max_horizon + 1))])
        else:
            self.test_dataset = None
    
    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)