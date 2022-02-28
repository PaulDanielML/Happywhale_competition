from typing import Tuple
import pandas as pd
from functools import partial
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from common import save_structure, load_structure, load_train_file, OUTPUT_DATA_DIR

load_split = partial(load_structure, path=OUTPUT_DATA_DIR / "splits")


def create_train_val_loaders(
    train_dataset: Dataset,
    valid_dataset: Dataset,
    train_batch_size: int = 32,
    valid_batch_size: int = 64,
    n_workers: int = 2,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=n_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        num_workers=n_workers,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, valid_loader


def split_into_train_val(
    df: pd.DataFrame, fold_number_for_val_set: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df[df["fold_number"] != fold_number_for_val_set].reset_index(drop=True)
    df_valid = df[df["fold_number"] == fold_number_for_val_set].reset_index(drop=True)

    return df_train, df_valid


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    enc = LabelEncoder()
    if "individual_id" not in df.columns:
        return df
    df = df.copy(deep=True)
    df["individual_id"] = enc.fit_transform(df["individual_id"])
    save_structure(enc, "id_encoding", OUTPUT_DATA_DIR)

    return df


def create_StratifiedKFold_split(
    df: pd.DataFrame = None, k: int = 5, save: bool = True, encode_targets: bool = True
) -> pd.DataFrame:
    if df is None:
        if encode_targets:
            df = encode_labels(load_train_file())
        else:
            df = load_train_file()
    skf = StratifiedKFold(k)
    for fold_number, (_, test_idx) in enumerate(skf.split(X=df, y=df.individual_id)):
        df.loc[test_idx, "fold_number"] = fold_number

    if save:
        save_structure(df, f"stratified_{k}_fold", OUTPUT_DATA_DIR / "splits")
    return df


def create_idvs_with_one_img_in_train_split(save: bool = True) -> pd.DataFrame:
    df = encode_labels(load_train_file())
    idvs_with_only_one_train_image = (
        df["individual_id"].value_counts().to_frame().query("individual_id == 1").index.tolist()
    )
    mask = df["individual_id"].isin(idvs_with_only_one_train_image)
    df_split = create_StratifiedKFold_split(df.loc[~mask].reset_index(drop=True), k=4, save=False)
    df_final = pd.concat([df.loc[mask], df_split])
    df_final["fold_number"].fillna("idv_with_one_img", inplace=True)

    if save:
        save_structure(df_final, "idvs_with_one_img_in_train", OUTPUT_DATA_DIR / "splits")
    return df_final


if __name__ == "__main__":
    print(load_split("5"))
