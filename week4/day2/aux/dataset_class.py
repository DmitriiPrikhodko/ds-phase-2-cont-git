import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MovieLensDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        user_col: int = 0,
        item_col: int = 1,
        rating_col: int = 2,
    ):

        self.df = df

        self.user_tensor = torch.tensor(
            self.df.iloc[:, user_col].to_numpy(), dtype=torch.long, device="cpu"
        )
        self.item_tensor = torch.tensor(
            self.df.iloc[:, item_col].to_numpy(), dtype=torch.long, device="cpu"
        )
        self.trgt_tensor = torch.tensor(
            self.df.iloc[:, rating_col].to_numpy(), dtype=torch.float32, device="cpu"
        )

    def __len__(self):
        return self.user_tensor.shape[0]
        # return len(self.df)

    def __getitem__(self, index):
        return (
            self.user_tensor[index],
            self.item_tensor[index],
            self.trgt_tensor[index],
        )
