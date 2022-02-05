import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src import DATA_DIR


def show_images(df: pd.DataFrame, columns: int = 4):
    is_train = "individual_id" in df.columns
    _dir_name = "train_images" if is_train else "test_images"
    img_dir = DATA_DIR / _dir_name
    rows = len(df) // columns
    if len(df) % columns != 0:
        rows += 1

    plt.figure(figsize=(columns * 5, rows * 5))

    for i, row in df.reset_index(drop=True).iterrows():
        plt.subplot(rows, columns, i + 1)
        try:
            img = np.array(Image.open(img_dir / row.image))
        except FileNotFoundError:
            print(f"{row.image}: not found")
            continue
        plt.imshow(img)
        if is_train:
            plt.title(f"{row.species}\nID: {row.individual_id}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
