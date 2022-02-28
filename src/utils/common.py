from pathlib import Path
import pickle
from typing import Any
import os
import pandas as pd
import numpy as np
import torch

WANDB_KEY = "3f26adf9c93e22c2a1316b3781fceca72e093ad4"

ON_KAGGLE_KERNEL = os.path.isdir("/kaggle/input")

if ON_KAGGLE_KERNEL:
    BASE_DIR = Path("/kaggle/working")
    if Path("/kaggle/input/happywhale-cropped").is_dir():
        INPUT_DATA_DIR = Path("/kaggle/input/happywhale-cropped")
    else:
        INPUT_DATA_DIR = Path("/kaggle/input/happy-whale-and-dolphin")

    WEIGHTS_DIR = Path("/kaggle/input/arcface-weights")
else:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    INPUT_DATA_DIR = BASE_DIR / "data" / "cropped"
    # INPUT_DATA_DIR = BASE_DIR / "data"

    WEIGHTS_DIR = Path("kernels/arcface/weights")

OUTPUT_DATA_DIR = BASE_DIR / "data"
SUB_DIR = BASE_DIR / "submissions"


def load_train_file() -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(INPUT_DATA_DIR / "train.csv")  # type: ignore
    df["species"].replace(
        {
            "globis": "short_finned_pilot_whale",
            "pilot_whale": "short_finned_pilot_whale",
            "kiler_whale": "killer_whale",
            "bottlenose_dolpin": "bottlenose_dolphin",
        },
        inplace=True,
    )
    df["img_path"] = df["image"].apply(lambda x: str(INPUT_DATA_DIR / "train_images" / x))
    return df


def load_test_file() -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(INPUT_DATA_DIR / "sample_submission.csv")  # type: ignore
    df["img_path"] = df["image"].apply(lambda x: str(INPUT_DATA_DIR / "test_images" / x))

    return df.drop(columns="predictions")


def map_per_image(label, predictions):
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0


def match_search_str_in_dir(search_str: str, dir: Path) -> Path:
    matching_files = [f for f in os.listdir(dir) if (dir / f).is_file() and search_str in f]
    if len(matching_files) > 1:
        if len(exact_match := [f for f in matching_files if f.split(".")[0] == search_str]) == 1:
            return dir / exact_match[0]
        raise ValueError(f"More than one match found for search string {search_str}.")
    if not matching_files:
        raise ValueError(f"No matches found for search string {search_str} (looking in {dir}).")

    return dir / matching_files[0]


def save_structure(
    obj: Any, name: str, path: Path = OUTPUT_DATA_DIR, overwrite: bool = True
) -> None:
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)

    file_path = path / f"{name}.pickle"

    if file_path.is_file() and not overwrite:
        print(f"Not saving {file_path}!")
        return
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

    print(f"{file_path} saved.")


def load_structure(name: str, path: Path = OUTPUT_DATA_DIR) -> None:
    with open(path / match_search_str_in_dir(name, path), "rb") as f:
        return pickle.load(f)


def create_dir_if_not_exists(path: Path) -> None:
    if not path.is_dir():
        try:
            path.mkdir(parents=True)
        except Exception as e:
            print(repr(e))


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    print(load_train_file())
