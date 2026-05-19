# download database from Kaggle and rename it to mushroom_data.zip to use

import zipfile
from pathlib import Path
import random
import shutil

dataset_zip = "mushroom_data.zip"
destination = Path("organized_mushroom_data/")
possible_files = ('.jpg', '.jpeg', '.png')

# delete organized data and recreate it
if destination.exists():
    confirmation = input("Are you sure you want to delete the current organized data and recreate it? y/n ")
    if confirmation == "y":
        shutil.rmtree(destination)
        
categorized_images = {
    'conditionally_edible': [],
    'deadly': [],
    'edible': [],
    'poisonous': [],
}

# Sort the images into their main folders
with zipfile.ZipFile(dataset_zip, 'r') as archive:
    #get all files and directories in a zip
    all_files = archive.namelist()

    for file_path in all_files:
        if not file_path.endswith('/') and file_path.endswith(possible_files):
            # split the file
            file_parts = file_path.split('/')
            main_folder = file_parts[2]

            categorized_images[main_folder].append(file_path)
    
    # Extract 2,000 images from each folder
    for category, images in categorized_images.items():
        # Pick 2,000 random files
        file_amount = min(len(images), 1000)
        selected_files = random.sample(images, file_amount)

        # Create new organized folder
        category_destination = destination / category
        category_destination.mkdir(parents=True, exist_ok=True)

        # Extract file and flatten them
        for index, file_path in enumerate(selected_files):
            image_data = archive.read(file_path)

            extension = Path(file_path).suffix
            unique_name = f"{category}_{index}{extension}"

            # New path
            new_file_path = category_destination / unique_name

            # Write it into each main folder
            with open(new_file_path, 'wb') as new_file:
                new_file.write(image_data)
