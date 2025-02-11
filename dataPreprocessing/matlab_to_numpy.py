import os
import h5py
import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

LABELS_MAP = {1: "meningioma", 2: "glioma", 3: "pituitary_tumor"}

def process_mat_files(data_dir, output_dir):
    """Process all .mat files in the data directory and save images + labels."""
    
    os.makedirs(output_dir, exist_ok=True) 
    csv_data = []  

    mat_files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
    
    for mat_file in tqdm(mat_files, desc="Processing .mat files"):
        file_path = os.path.join(data_dir, mat_file)

        try:
            with h5py.File(file_path, "r") as f:
                if "cjdata" not in f:
                    print(f"Skipping {mat_file}: 'cjdata' not found.")
                    continue
                
                cjdata = f["cjdata"]

                label = int(cjdata["label"][0, 0]) 
                image = np.array(cjdata["image"])   
                
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

                img_filename = mat_file.replace(".mat", ".png")
                img_path = os.path.join(output_dir, img_filename)

                cv2.imwrite(img_path, image)

                csv_data.append([img_filename, LABELS_MAP[label]])

        except Exception as e:
            print(f"Error processing {mat_file}: {e}")
    
    df = pd.DataFrame(csv_data, columns=["filename", "tumor_type"])
    df.to_csv(os.path.join(output_dir, "data_labels.csv"), index=False)
    print(f"\nSaved {len(csv_data)} images and data_labels.csv in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Brain Tumor .mat files to PNG and CSV")
    parser.add_argument("data_dir", type=str, help="Path to the directory containing .mat files")
    parser.add_argument("output_dir", type=str, help="Path to the output directory for images and CSV")
    
    args = parser.parse_args()
    process_mat_files(args.data_dir, args.output_dir)