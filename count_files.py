from pathlib import Path

def check_folder_counts(target_directory):
    main_folder = Path(target_directory)
        
    # Look at every file in main folder
    for item in main_folder.iterdir():
        
        # Look only inside sub-folders
        if item.is_dir():        
            files_inside = []            
            for file in item.iterdir():
                if file.is_file():
                    files_inside.append(file)
            
            print(f"{item.name}: {len(files_inside)} files")

check_folder_counts("organized_mushroom_data")