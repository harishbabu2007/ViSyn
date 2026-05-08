import os

def rename_images(folder_path):
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
    
    # Get all files and sort them alphabetically
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    files.sort()
    
    for index, filename in enumerate(files, start=1):
        # Keep original extension (e.g., .jpg)
        extension = os.path.splitext(filename)[1]
        
        # Define old and new paths
        old_path = os.path.join(folder_path, filename)
        new_name = f"{index}{extension}"
        new_path = os.path.join(folder_path, new_name)
        
        # Rename the file
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
        except FileExistsError:
            print(f"Error: {new_name} already exists. Skipping.")


rename_images('C:\\Code\\ViSyn\\train_dataset\\inputs')
