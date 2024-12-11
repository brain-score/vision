import os
import re

def process_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Replace bucket="brainio-brainscore" with the new bucket string (accounting for variations)
    content = re.sub(
        r'"brainio-brainscore"',
        r'"brainscore_storage/brainio-brainscore"',
        content
    )


    # Replace csv_version_id and zip_version_id with "null", including complex hash-like values
    content = re.sub(r'csv_version_id="[\w.-]+"', 'csv_version_id="null"', content)
    content = re.sub(r'zip_version_id="[\w.-]+"', 'zip_version_id="null"', content)

    # Write the changes back to the file if any modifications were made
    with open(file_path, 'w') as file:
        file.write(content)

    print(f"Processed: {file_path}")

def process_folder(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for file_name in files:
            if file_name.endswith('.py'):
                file_path = os.path.join(root, file_name)
                process_file(file_path)

if __name__ == "__main__":
    root_folder = input("Enter the root folder path: ")
    process_folder(root_folder)