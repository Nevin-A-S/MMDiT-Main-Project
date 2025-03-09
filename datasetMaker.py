import os
import pandas as pd

base_dir = "dataPreprocessing"
captions = []
image = []

for files in os.listdir(base_dir):
    if files.endswith('.txt'):
        with open(os.path.join(base_dir,files)) as f:
            s = f.read()
            captions.append(s)
        file = files[:-4]
        file = file [:-7]
        file = file + '.png'
        print(file)
        image.append((os.path.join(base_dir,files)))


dataframe  = pd.DataFrame(image,captions)
dataframe.to_csv("synthetic_dataset.csv", index=False)