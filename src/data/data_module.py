import pandas as pd

from util.config import load_config


class DataModule:
    def __init__(self, config_path, task, x_col="sentence", y_col="label"):
        cf = load_config(config_path)
        self.train = pd.read_csv(cf[task]["train_dataset"])
        self.val = pd.read_csv(cf[task]["val_dataset"])

        self.train_size = len(self.train)
        self.val_size = len(self.val)
        self.size = self.train_size + self.val_size
        
        self.x_col = x_col
        self.y_col = y_col

        self.num_classes = len(
            set(self.train[y_col].unique()).union(
                set(self.val[y_col].unique())
            )
        )

    def _iter_xy(self, df):
        x = df[self.x_col].to_numpy(copy=False)
        y = df[self.y_col].to_numpy(copy=False)
        return zip(x, y)

    def iter_train(self):
        return self._iter_xy(self.train)

    def iter_val(self):
        return self._iter_xy(self.val)

    def iter_all(self):
        return self._iter_xy(pd.concat([self.train, self.val], ignore_index=True))

