import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
from ml.dataset import XView2BuildingDataset

metadata_path = "data/xview2_metadata.csv"

dataset = XView2BuildingDataset(metadata_path)

print("Total samples:", len(dataset))

shown = 0
index = 0

while shown < 5 and index < len(dataset):

    sample = dataset[index]
    index += 1

    if sample is None:
        continue

    image, label = sample
    print(f"Sample shape: {image.shape}, Label: {label}")

    image, label = sample
    
    # Image is now the difference
    diff = image

    plt.imshow(diff)
    plt.title(f"Difference (Label: {label})")

    plt.suptitle(f"Damage Class: {label}")
    plt.show()

    shown += 1