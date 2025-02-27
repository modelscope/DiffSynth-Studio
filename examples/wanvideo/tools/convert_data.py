import os
import pandas as pd
import shutil

# Define source and target directories
source_dir = "ori_data/halle"  # Source directory
output_dir = "data/halle"
train_dir = os.path.join(output_dir, "train")
metadata_file = os.path.join(output_dir, "metadata.csv")

# Create the target directory structure
os.makedirs(train_dir, exist_ok=True)

# Initialize metadata.csv data
metadata = []

# Get all files in the source directory
files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Process media files (videos, images) and their corresponding text files
media_count = 0  # Counter for media files (videos and images)
supported_media_extensions = [".mp4", ".jpg", ".png"]  # Supported media extensions

for file in sorted(files):
    # Check if the file is a supported media file (video or image)
    if any(file.lower().endswith(ext) for ext in supported_media_extensions):
        media_count += 1
        # Generate a new file name based on the file extension
        ext = file.rsplit(".", 1)[1].lower()
        new_media_name = f"media_{media_count:05d}.{ext}"
        new_media_path = os.path.join(train_dir, new_media_name)

        # Copy the media file to the target directory
        shutil.copy(os.path.join(source_dir, file), new_media_path)

        # Look for the corresponding text file
        text_file = file.rsplit(".", 1)[0] + ".txt"
        text_path = os.path.join(source_dir, text_file)
        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as f:
                # Read the text content and filter out empty lines
                text_content = "\n".join([line.strip() for line in f if line.strip()])
            # Add data to the metadata list
            metadata.append([new_media_name, text_content])
        else:
            # If no corresponding text file is found, record an empty description
            metadata.append([new_media_name, ""])

# Write metadata to metadata.csv
df = pd.DataFrame(metadata, columns=["file_name", "text"])
os.makedirs(output_dir, exist_ok=True)
df.to_csv(metadata_file, index=False)

print(f"Files have been successfully organized into {output_dir}")
