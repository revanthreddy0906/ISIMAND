import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from shapely.geometry import Polygon
import json


class XView2BuildingDataset:
    def __init__(self, metadata_csv, patch_size=128):
        self.df = pd.read_csv(metadata_csv)
        self.patch_size = patch_size

    def __len__(self):
        return len(self.df)

    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def crop_building(self, image, polygon_coords):
        poly = Polygon(polygon_coords)
        minx, miny, maxx, maxy = map(int, poly.bounds)

        crop = image[miny:maxy, minx:maxx]

        if crop.size == 0:
            return None

        crop = cv2.resize(crop, (self.patch_size, self.patch_size))
        return crop

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pre_img = self.load_image(row["pre_image_path"])
        post_img = self.load_image(row["post_image_path"])

        # Load polygon from JSON
        with open(row["json_path"]) as f:
            data = json.load(f)

        building = data["features"]["xy"][row["building_id"]]
        polygon = building["geometry"]["coordinates"][0]

        pre_crop = self.crop_building(pre_img, polygon)
        post_crop = self.crop_building(post_img, polygon)

        if pre_crop is None or post_crop is None:
            return None

        # Stack to 6 channels
        stacked = np.concatenate([pre_crop, post_crop], axis=-1)

        label = row["damage_class"]

        return stacked.astype(np.float32) / 255.0, label