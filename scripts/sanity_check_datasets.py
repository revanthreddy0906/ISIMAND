import os 
import random
import matplotlib.pyplot as plt
from PIL import Image

base_dir = "data/processed"
classes = ["no_damage","moderate_damage","severe_damage"]

sample_per_class = 5

for cls in classes:
    class_dir = os.path.join(base_dir,cls)
    if not os.path.exists(class_dir):
        print(f"[WARN!!!] Directory not found: {class_dir}")
        continue
    image_files = os.listdir(class_dir)
    if len(image_files) == 0:
        print(f"[WARN!!!] No images found in {class_dir}")
        continue

    print(f"Displaying Samples from : {cls}")
    samples = random.sample(image_files,min(sample_per_class,len(image_files)))
    plt.figure(figsize=(15,3))
    plt.suptitle(f"Class: {cls} ({len(image_files)} images)",fontsize=16)
    for i,file in enumerate(samples):
        img_path = os.path.join(class_dir,file)
        img = Image.open(img_path)
        plt.subplot(1,sample_per_class,i+1)
        plt.imshow(img)
        plt.title(file[:10])
        plt.axis("off")
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()