import os
import pandas as pd

base_dir = "test_folder"
captions = []
images = []

for file in os.listdir(base_dir):
    if file.endswith('.txt'):
        with open(os.path.join(base_dir, file), "r") as f:
            caption = f.read().strip()
            captions.append(caption)

        image_filename = file.replace("_prompt.txt", ".png")  
        images.append(os.path.join(base_dir, image_filename))

dataframe = pd.DataFrame({'Image': images, 'Caption': captions})

print(dataframe)
dataframe.to_csv("synthetic_dataset.csv", index=False)
