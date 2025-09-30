import pandas as pd
import numpy as np
import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


def recommend_for_user_id_from_valid(
    user_id: int,
    valid_df: pd.DataFrame,  # type: ignore
    model: torch.nn.Module,
    device,
    id_to_movie: dict,
    n=-1,
):

    user_df = valid_df[valid_df["user_id"] == user_id].copy()
    # print(user_df.head(3))
    if user_df.shape[0] == 0:
        raise IndexError("There is no user with this ID in valid dataset")

    user_tensor = torch.tensor(
        user_df["user_id"].to_numpy(), dtype=torch.long, device=device
    )
    item_tensor = torch.tensor(
        user_df["item_id"].to_numpy(), dtype=torch.long, device=device
    )
    # print(user_tensor)
    # print(item_tensor)
    model.eval()
    with torch.inference_mode():
        preds = model(user_tensor, item_tensor)
    # print(preds)
    # print(preds.squeeze(-1))
    # print(preds.squeeze(-1).cpu().numpy())
    user_df["pred_rating"] = preds.squeeze(-1).cpu().numpy()
    user_df["title"] = user_df["item_id"].apply(lambda x: id_to_movie[x])
    user_df.rename(columns={"item_id": "movie_id"}, inplace=True)
    user_df_sorted = user_df.loc[
        :, ["movie_id", "title", "pred_rating", "rating"]
    ].sort_values("pred_rating", ascending=False)
    if n == -1:  # Возвращаем все фильмы
        return user_df_sorted
    else:  # Возвращаем топ n по предсказанной оценке
        return user_df_sorted.iloc[:n, :]


def recommend_for_user_id_unwatched(
    user_id: int,
    full_df: pd.DataFrame,  # type: ignore
    model: torch.nn.Module,
    device,
    id_to_movie: dict,
    n=-1,
):

    user_watched = full_df.copy()[full_df["user_id"] == user_id]

    watched = user_watched["item_id"].to_list()

    unwatched = {}

    for idx in id_to_movie.keys():
        if idx not in watched:
            unwatched[idx] = id_to_movie[idx]

    unwatched_df = pd.DataFrame(list(unwatched.items()), columns=["movie_id", "title"])

    unwatched_df["user_id"] = np.array([user_id for _ in range(len(unwatched.keys()))])
    user_tensor = torch.tensor(
        unwatched_df["user_id"].to_numpy(), dtype=torch.long, device=device
    )
    item_tensor = torch.tensor(
        unwatched_df["movie_id"].to_numpy(), dtype=torch.long, device=device
    )

    model.eval()
    with torch.inference_mode():
        preds = model(user_tensor, item_tensor)

    unwatched_df["pred_rating"] = preds.squeeze(-1).cpu().numpy()
    unwatched_df_sorted = unwatched_df.loc[
        :, ["movie_id", "title", "pred_rating"]
    ].sort_values("pred_rating", ascending=False)
    if n == -1:  # Возвращаем все фильмы
        return (
            unwatched_df_sorted,
            user_watched.shape[0],
            len(id_to_movie.keys()),
        )
    else:  # Возвращаем топ n по предсказанной оценке
        return (
            unwatched_df_sorted.iloc[:n, :],
            user_watched.shape[0],
            len(id_to_movie.keys()),
        )
