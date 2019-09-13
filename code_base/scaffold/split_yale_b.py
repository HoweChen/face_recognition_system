import pathlib
import os

yale_b_dir = pathlib.Path("/Users/howechen/Dropbox/Lab/dataset/CV/CroppedYale")
classifier_yale_b_dir = pathlib.Path("/Users/howechen/Dropbox/Lab/dataset/CV/classified_yale_b")

# same vertical_degree
same_vertical_folder_path = str(classifier_yale_b_dir.absolute()) + "/" + "same_vertical_degree"
print(same_vertical_folder_path)
os.system(f"mkdir -p {same_vertical_folder_path}")

# same horizontal_degree
same_horizontal_folder_path = str(classifier_yale_b_dir.absolute()) + "/" + "same_horizontal_degree"
print(same_horizontal_folder_path)
os.system(f"mkdir -p {same_horizontal_folder_path}")
range_lists = [range(0,)]

for dir in yale_b_dir.iterdir():
    if dir.is_dir():
        for image in dir.iterdir():
            if image.name[-12:-4] == "+000E+00":
                os.system(f"cp {str(image.absolute())} {str(classifier_yale_b_dir.absolute())}")
            if image.name[-7:-4] == "+00":
                os.system(f"cp {str(image.absolute())} {same_vertical_folder_path}")
            if image.name[-12:-7] == "+000E":
                os.system(f"cp {str(image.absolute())} {same_horizontal_folder_path}")

