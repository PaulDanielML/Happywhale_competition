from ctypes import Union
from typing import Dict, Any, List, Type, Optional
import copy
import os
import datetime
import sys
import subprocess
import time

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler, Optimizer, Adam
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from prettytable import PrettyTable
from pydantic import BaseModel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

from common import (
    INPUT_DATA_DIR,
    OUTPUT_DATA_DIR,
    load_structure,
    save_structure,
    load_test_file,
    map_per_image,
    SUB_DIR,
)
from data_proc import create_StratifiedKFold_split


PLOTLY_DEF_LAYOUT = go.Layout(
    width=1000,
    height=600,
    title=dict(x=0.5, y=0.95, font=dict(family="Arial", size=25, color="#000000")),
    font=dict(family="Courier New, Monospace", size=15, color="#000000"),
)


class TorchConfig(BaseModel):
    seed: int
    epochs: int
    img_size: int
    augm_args: Dict[str, Dict[str, Any]]
    model_name: str
    num_classes: int
    train_batch_size: int
    valid_batch_size: int
    optim: Type[Optimizer]
    optim_args: Dict[str, Any]
    scheduler: Type[lr_scheduler._LRScheduler]
    scheduler_args: Dict[str, Any]
    n_fold: int

    init_optim_: Optional[Optimizer] = None
    init_sched_: Optional[lr_scheduler._LRScheduler] = None

    # ArcFace Hyperparameters
    s = 30.0
    m = 0.5
    ls_eps = 0.0
    easy_margin = False

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def default(cls):
        return cls(
            seed=319,
            epochs=10,
            img_size=448,
            augm_args={
                "hor_flip": {"p": 0.5},
                "ver_flip": {"p": 0.5},
                "rot": {"p": 0.5, "limit": 30},
            },
            model_name="tf_efficientnet_b0",
            num_classes=15587,
            train_batch_size=32,
            valid_batch_size=64,
            optim=Adam,
            optim_args={"lr": 1e-4, "weight_decay": 1e-6},
            scheduler=lr_scheduler.CosineAnnealingLR,
            scheduler_args={"T_max": 500, "eta_min": 1e-6},
            n_fold=5,
        )

    def get_optim(self, model: nn.Module):
        self.init_optim_ = self.optim(model.parameters(), **self.optim_args)
        return self.init_optim_

    def get_scheduler(self):
        if self.init_optim_ is None:
            raise ValueError("Need to init optimizer first.")
        self.init_sched_ = self.scheduler(optimizer=self.init_optim_, **self.scheduler_args)
        return self.init_sched_

    def make_transforms(
        self,
    ):
        param_mapping = {
            "hor_flip": A.HorizontalFlip,
            "ver_flip": A.VerticalFlip,
            "rot": A.Rotate,
        }

        base_transforms = [
            A.Resize(self.img_size, self.img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]

        train_transforms = copy.deepcopy(base_transforms)

        for aug_key, params in self.augm_args.items():
            aug_type = param_mapping[aug_key]
            if "p" in params.keys() and params["p"] <= 0.0:
                continue
            train_transforms.insert(1, aug_type(**params))

        return {"train": A.Compose(train_transforms), "valid": A.Compose(base_transforms)}

    def __repr__(self):
        table = PrettyTable(field_names=["Parameter", "Value"], hrules=1, title="CONFIG")
        for k, v in self.__dict__.items():
            if k.endswith("_"):
                continue
            table.add_row([k, v])
        print(table)
        return ""


class WhaleDataset(Dataset):
    """
    Generic dataset template. Returns dict of transformed images (key: 'image') and target values
    (key: 'label'), if param labels = True.
    If labels = False, it is assumed that the intended use is inference on test_images.
    """

    def __init__(
        self, df: pd.DataFrame, transforms: Optional[A.Compose] = None, labels: bool = True
    ):
        self.df = df.copy(deep=True)
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = np.array(Image.open(row.img_path).convert("RGB"))

        if self.transforms:
            img = self.transforms(image=img)["image"]

        data = {"image": img}
        if not self.labels:
            return data
        data["label"] = torch.tensor(row.individual_id, dtype=torch.long)
        return data


FutureArray = Optional[np.ndarray]
FutureDf = Optional[pd.DataFrame]


class InferenceKNNModel:

    emb_dir = OUTPUT_DATA_DIR / "embeddings"

    def __init__(self, model: nn.Module = None, name: str = None, num_folds: int = 5):
        self.model = model
        if model is not None:
            self.model.eval()

        if name is None:
            self.name = datetime.datetime.now().strftime("%y%m%d-%H_%M")
        else:
            self.name = name

        self.train_data = create_StratifiedKFold_split(save=False, encode_targets=False)
        self.test_data = load_test_file()
        self.calc_data: Dict[str, Union[FutureArray, FutureDf]] = {  # type: ignore
            "train_emb": None,
            "test_emb": None,
            "tsne": None,
            "knn_dist": None,
            "knn_dist_norm": None,
            "knn_idx": None,
            "knn_metric": None,
            "knn_thr": None,
            "thr_normalized": False,
        }
        self.num_folds = num_folds
        self.continue_previous = True
        self.train_start_idx, self.test_start_idx = 0, 0
        self.transforms = TorchConfig.default().make_transforms()["valid"]
        self.emb_dir.mkdir(parents=True, exist_ok=True)

        self.load_data()
        SUB_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def train_emb(self):
        return self.calc_data["train_emb"]

    @property
    def test_emb(self):
        return self.calc_data["test_emb"]

    @property
    def dist_threshold(self):
        return self.calc_data["knn_thr"]

    @property
    def train_dataset(self):
        return WhaleDataset(self.train_data[self.train_start_idx :], self.transforms, labels=False)

    @property
    def test_dataset(self):
        return WhaleDataset(self.test_data[self.test_start_idx :], self.transforms, labels=False)

    def get_neighbors(
        self, n: int = 30, distance_metric: str = "euclidean", force_rerun: bool = False
    ):
        knn = NearestNeighbors(metric=distance_metric)

        if (not self._check_calc_done("train_emb")) or (not self._check_calc_done("test_emb")):
            self.generate_embeddings()

        if (
            force_rerun
            or (self.calc_data.get("knn_dist") is None)
            or (self.calc_data.get("knn_dist_norm") is None)
            or (self.calc_data.get("knn_dist").shape[1] != n)
            or (self.calc_data.get("knn_metric") != distance_metric)
        ):
            knn.fit(self.train_emb)
            self.calc_data["knn_dist"], self.calc_data["knn_idx"] = knn.kneighbors(self.test_emb, n)

            self.calc_data["knn_dist_norm"] = (
                MinMaxScaler()
                .fit_transform(self.calc_data["knn_dist"].reshape(-1, 1))
                .reshape(self.calc_data["knn_dist"].shape)
            )

            self.calc_data["knn_metric"] = distance_metric
            self.save_data()

    @staticmethod
    def get_predictions(distances, neighbors, threshold, idx_target_mapper):
        num_neighbors = neighbors.shape[1]
        all_predictions = []

        for img_dist, img_neigh in zip(distances, neighbors):
            new_idv_inserted = False
            predictions = []
            for i in range(num_neighbors):
                if len(predictions) == 5:
                    break
                idv = idx_target_mapper[img_neigh[i]]
                if (img_dist[i] < threshold) or new_idv_inserted:
                    if idv not in predictions:
                        predictions.append(idv)
                    continue
                predictions.append("new_individual")
                if img_neigh[i] not in predictions:
                    predictions.append(idv)
                new_idv_inserted = True
            all_predictions.append(predictions)

        return all_predictions

    def tune_threshold(
        self,
        distance_metric: str = "euclidean",
        thresholds_to_try: List = [0.1 * x for x in range(8)],
        force_rerun: bool = False,
        normalize_distances: bool = True,
    ):
        if (self.calc_data.get("knn_thr") is not None) and (not force_rerun):
            return

        def _get_fold_val_scores_by_thr(df_train: pd.DataFrame, df_val: pd.DataFrame):
            knn = NearestNeighbors(metric=distance_metric)

            knn.fit(self.train_emb[df_train.index])
            dist, idx = knn.kneighbors(self.train_emb[df_val.index], 30)

            if normalize_distances:
                dist = MinMaxScaler().fit_transform(dist.reshape(-1, 1)).reshape(dist.shape)

            threshold_scores = []
            for threshold in thresholds_to_try:

                predictions = self.get_predictions(
                    dist, idx, threshold, df_train.reset_index()["individual_id"].to_dict()
                )
                scores = [
                    map_per_image(df_val.iloc[i]["individual_id"], pred)
                    for i, pred in enumerate(predictions)
                ]
                threshold_scores.append(np.array(scores).mean())

            return threshold_scores  # Scores of each threshold for that split

        fold_scores = []

        if normalize_distances:
            self.calc_data["thr_normalized"] = True
        else:
            self.calc_data["thr_normalized"] = False

        for i in range(self.num_folds):
            fold_mask = self.train_data["fold_number"] == i
            df_train = self.train_data.loc[~fold_mask]
            df_val = self.train_data.loc[fold_mask]

            train_idvs = df_train["individual_id"].unique()
            new_idvs_mask = ~df_val["individual_id"].isin(train_idvs)
            df_val.loc[new_idvs_mask, "individual_id"] = "new_individual"

            fold_scores.append(_get_fold_val_scores_by_thr(df_train, df_val))

        fold_scores = np.array(fold_scores)

        mean_scores = fold_scores.mean(axis=0)
        best_score = mean_scores.max()
        best_threshold = thresholds_to_try[mean_scores.argmax()]

        print(
            f"CV to find threshold finished. Best threshold: {best_threshold}, best score: {best_score}."
        )

        self.calc_data["knn_thr"] = best_threshold

        self.save_data()

    def create_test_predictions(self, create_subm_file: bool = True):
        predictions = self.get_predictions(
            self.calc_data["knn_dist"]
            if not self.calc_data["thr_normalized"]
            else self.calc_data["knn_dist_norm"],
            self.calc_data["knn_idx"],
            self.dist_threshold,
            self.train_data["individual_id"].to_dict(),
        )

        pred_str = [" ".join(p) for p in predictions]

        df_subm = self.test_data.copy(deep=True)
        df_subm["predictions"] = pred_str
        df_subm.set_index("image", inplace=True)

        if create_subm_file:
            df_subm["predictions"].to_csv(SUB_DIR / f"sub_{self.name}.csv")
        return df_subm

    def _make_loader(self, train_or_test: str):
        dataset = self.train_dataset if train_or_test == "train" else self.test_dataset
        if not dataset:
            return
        return DataLoader(
            dataset,
            32,
            shuffle=False,
            num_workers=6,
        )

    @staticmethod
    def list_all():
        return {f.split(".")[0] for f in os.listdir(InferenceKNNModel.emb_dir)}

    def end_to_end(
        self,
        recalc_emb: bool = False,
        recalc_neigh: bool = False,
    ):
        self.generate_embeddings(force_rerun=recalc_emb)
        self.tsne()
        self.get_neighbors(force_rerun=(recalc_neigh or recalc_emb))
        self.tune_threshold(force_rerun=(recalc_neigh or recalc_emb))
        self.create_test_predictions()
        self.make_submission()
        self.show_submission_results()

    def generate_embeddings(self, force_rerun: bool = False):
        if force_rerun:
            self.continue_previous = False

        self._set_start_idx()

        self._get_embeddings("train")
        self._get_embeddings("test")

    def save_data(self):
        save_structure(self.calc_data, self.name, self.emb_dir)

    def load_data(self, name: Optional[str] = None):
        try:
            self.calc_data.update(load_structure(name or self.name, self.emb_dir))
            print(f"Embeddings loaded from {self.emb_dir}/{name or self.name}")
        except ValueError:
            print(f"No embeddings found for {self.name}")

    def tsne(self, species: Optional[str] = None):
        if self._check_calc_done("tsne"):
            return self._plot_tsne(species)

        if (not self._check_calc_done("train_emb")) or (not self._check_calc_done("test_emb")):
            self.generate_embeddings()

        df_combined = pd.concat(
            [self.train_data.assign(train=True), self.test_data.assign(train=False)]
        ).reset_index()

        df_combined["species"] = df_combined.apply(
            lambda x: x["species"] if x["train"] else "Test", axis=1
        )

        df_combined["Category"] = df_combined[["train", "individual_id"]].apply(
            lambda x: (f'Train - {x["individual_id"]}' if x["train"] else "Test"), axis=1
        )

        emb_combined = np.concatenate([self.train_emb, self.test_emb])

        assert df_combined.shape[0] == emb_combined.shape[0]

        pc = PCA(n_components=50).fit_transform(emb_combined)

        tsne = TSNE(learning_rate="auto", verbose=1, init="pca").fit_transform(pc)
        tsne_columns = ["tsne_0", "tsne_1"]
        df_combined[tsne_columns] = tsne

        self.calc_data["tsne"] = df_combined

        self.save_data()

        return self._plot_tsne(species)

    def _plot_tsne(self, species: Optional[str]):
        print(
            f'Possible species values are:\n{self.calc_data["tsne"]["species"].unique().tolist()}'
        )

        if species is None:
            return px.scatter(
                self.calc_data["tsne"], x="tsne_0", y="tsne_1", color="species", title="All species"
            ).update_layout(PLOTLY_DEF_LAYOUT)

        df_plot = self.calc_data["tsne"].query("species == @species")
        return px.scatter(
            df_plot, x="tsne_0", y="tsne_1", color="Category", title=species
        ).update_layout(PLOTLY_DEF_LAYOUT)

    def _append_to_emb(self, data: np.ndarray, train_or_test: str):
        if self.calc_data[f"{train_or_test}_emb"] is None:
            self.calc_data[f"{train_or_test}_emb"] = data
        else:
            self.calc_data[f"{train_or_test}_emb"] = np.concatenate(
                [self.calc_data[f"{train_or_test}_emb"], data], axis=0
            )

    def _set_start_idx(self):
        if not self.continue_previous:
            return
        if self.calc_data["train_emb"] is None:
            self.train_start_idx = 0
        else:
            print(f"Found previous train embeddings for {self.name}.")
            self.train_start_idx = self.calc_data["train_emb"].shape[0]
        if self.calc_data["test_emb"] is None:
            self.test_start_idx = 0
        else:
            print(f"Found previous test embeddings for {self.name}.")
            self.test_start_idx = self.calc_data["test_emb"].shape[0]

    @torch.inference_mode()
    def _get_embeddings(self, train_or_test: str):
        loader = self._make_loader(train_or_test)
        if not loader:
            print(f"{train_or_test} embeddings: Already complete.")
            return
        if self.model is None:
            raise ValueError("Need model to generate embeddings.")

        def _finish(embeddings: List):
            self.model.cpu()
            data = np.concatenate(embeddings)
            self._append_to_emb(data, train_or_test)
            self.save_data()
            torch.cuda.empty_cache()

        if not next(self.model.parameters()).is_cuda:
            self.model.cuda()
        bar = tqdm(loader, total=len(loader), desc=f"Generating {train_or_test} embeddings")

        embeddings = []
        try:
            for batch in bar:
                imgs = batch["image"].cuda()
                emb = self.model(imgs)
                embeddings.append(emb.cpu().squeeze().numpy())
        except KeyboardInterrupt:
            del emb
            _finish(embeddings)
            sys.exit("Embeddings saved and exited.")

        del emb
        _finish(embeddings)

    def _check_calc_done(self, name: str):
        if self.calc_data.get(name) is None:
            return False
        expected_shapes = {
            "train_emb": self.train_data.shape[0],
            "test_emb": self.test_data.shape[0],
            "tsne": self.train_data.shape[0] + self.test_data.shape[0],
        }
        return self.calc_data[name].shape[0] == expected_shapes[name]

    def make_submission(self):
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "submit",
                "-f",
                str(SUB_DIR / f"sub_{self.name}.csv"),
                "-c",
                "happy-whale-and-dolphin",
                "-m",
                self.name,
            ]
        )
        time.sleep(5)

    @staticmethod
    def show_submission_results():
        subprocess.run(["kaggle", "competitions", "submissions", "happy-whale-and-dolphin"])


if __name__ == "__main__":
    test = InferenceKNNModel(name="ArcFace_first_try_uncropped")
    test.end_to_end()
    # test.get_neighbors(force_rerun=False)
    # test.tune_threshold(
    #     force_rerun=False, thresholds_to_try=[1 * x for x in range(10)], normalize_distances=False
    # )

    # test.create_test_predictions()
