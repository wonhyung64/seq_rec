#%%
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import requests
from io import BytesIO
from PIL import Image


# %%
root = os.getcwd()
data_dir = f"{root}/data/ver_23"

meta_df = pd.read_json(f"{data_dir}/meta_All_Beauty.jsonl", lines=True)
review_df = pd.read_json(f"{data_dir}/All_Beauty.jsonl", lines=True)
review_df = review_df[["rating", "parent_asin", "user_id", "timestamp", ]]

num_item = review_df["parent_asin"].nunique()
save_dir = "./eda"
os.makedirs(save_dir, exist_ok=True)

#%%
for i in tqdm(range(num_item)):
    if i > 9999: 
        break

    pop_item_idx = review_df["parent_asin"].value_counts().index[i]
    pop_item_df = review_df[review_df["parent_asin"] == pop_item_idx]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,4))
    axes[0].hist(pop_item_df["timestamp"], bins=100)
    axes[0].set_title(f"item: {pop_item_idx}")
    axes[0].set_ylabel("density")
    axes[0].tick_params(axis="x", labelrotation=45)

    url = meta_df[meta_df["parent_asin"] == pop_item_idx]["images"].item()[0]["large"]
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()
    img = Image.open(BytesIO(res.content)).convert("RGB")
    axes[1].imshow(img)
    axes[1].axis("off")
    plt.tight_layout()
    plt.close()
    
    fig.savefig(f"{save_dir}/item{str(i).zfill(4)}-{pop_item_idx}.png")
#%%
