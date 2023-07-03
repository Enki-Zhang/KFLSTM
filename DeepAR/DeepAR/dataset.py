import numpy as np
from sklearn.preprocessing import *
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):

    def minmaxscale(self, data: np.ndarray):
        seq_len, num_features = data.shape
        for i in range(num_features):
            min = data[:, i].min()
            max = data[:, i].max()
            data[:, i] = (data[:, i] - min) / (max - min)
        return data

    def __init__(self, params, scaler=StandardScaler()):
        import pandas as pd

        self.encode_step = params['encode_step']
        self.forcast_step = params['forcast_step']
        rawdata = pd.read_csv('pollution.csv')
        rawdata = rawdata.drop(labels=['date', 'wnd_dir', 'snow', 'rain'], axis=1)
        rawdata = rawdata.to_numpy().astype('float32')

        self.features = rawdata[:, 1:]
        self.label = rawdata[:, 0]

        self.features = self.minmaxscale(self.features)
        self.scaler = scaler
        self.scaler.fit(self.label.reshape(-1, 1))
        self.label = scaler.transform(self.label.reshape(-1, 1)).astype('float32')

    def __getitem__(self, index):
        # feature
        # [index + 1, t0] [t0 + 1, T]
        # lag input
        # [index, t0 - 1]

        # Step 1: lagged size adjust
        index += 1

        # Step 2: history features
        start = index
        end = start + self.encode_step
        hisx = self.features[start:end]

        # Step 3: history inputs
        hisz = self.label[start - 1:end - 1]

        # Step 4: future features
        start = end + 1
        end = start + self.forcast_step
        futx = self.features[start:end]

        # Step 5: targets
        z = self.label[index: index + self.encode_step + self.forcast_step]

        return hisx, hisz, futx, z

    def __len__(self):
        return len(self.features) - self.forcast_step - self.encode_step - 1
