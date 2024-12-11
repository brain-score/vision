import os
import re

def process_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Check if the target load_weight_file string is found
    if re.search(r'load_weight_file\(bucket="brainscore-vision", folder_name="data",', content):
        # Replace the load_weight_file string
        content = re.sub(
            r'load_weight_file\(bucket="brainscore-vision", folder_name="models",',
            r'load_weight_file(bucket="brainscore-storage", folder_name="brainscore-vision/models",',
            content
        )
        # Replace the version_id only if the above replacement was made
        content = re.sub(r'version_id=[\'"]?[\w-]+[\'"]?', 'version_id="null"', content)

        # Write the changes back to the file
        with open(file_path, 'w') as file:
            file.write(content)

        print(f"Processed: {file_path}")
    else:
        print(f"No match found in: {file_path}")

def process_folder(input_folder):
    for root, dirs, files in os.walk(input_folder):
        if 'model.py' in files:
            model_file_path = os.path.join(root, 'model.py')
            process_file(model_file_path)

if __name__ == "__main__":
    input_folder = input("Enter the path to the input folder: ")
    process_folder(input_folder)
