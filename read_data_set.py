import pandas as pd
from pathlib import Path
from skimage.io import imread

# --- CHANGE THESE PATHS TO MATCH YOUR FOLDERS ---
metadata_path = r"C:\Users\Dell\Desktop\MCA\Project Work\SkindiseaseProject\metadata.csv"
images_folder = Path(r"C:\Users\Dell\Desktop\MCA\Project Work\SkindiseaseProject\images")

# Load metadata
df = pd.read_csv(metadata_path)

# Print first 5 rows
print(df.head())

# Test reading ONE image
first_image_id = df.iloc[0]['image_id']   # example: ISIC_0024306
first_label = df.iloc[0]['dx']

image_path = images_folder / f"{first_image_id}.jpg"
print("Image path:", image_path)

# Load image
img = imread(str(image_path))
print("Image shape:", img.shape)
print("Label:", first_label)
