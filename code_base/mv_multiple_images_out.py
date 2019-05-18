import pathlib
import os

data_path = pathlib.Path("/Users/howechen/Dropbox/Lab/dataset/CV/lfw_multiple_images")

for dir in data_path.iterdir():
    if dir.is_dir():
        for image in dir.iterdir():
            os.system(f"mv {image.absolute()} /Users/howechen/Dropbox/Lab/dataset/CV/lfw_multiple_images")
print("Done")
