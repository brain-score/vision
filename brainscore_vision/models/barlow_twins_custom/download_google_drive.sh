#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <file_url> <destination_path>"
    exit 1
fi

file_url=$1
destination_path=$2

# Get the confirmation page
confirmation_page=$(curl -s -L "$file_url")

# Extract the necessary parameters
file_id=$(echo "$confirmation_page" | grep -oE "name=\"id\" value=\"[^\"]+" | sed 's/name="id" value="//')
file_confirm=$(echo "$confirmation_page" | grep -oE "name=\"confirm\" value=\"[^\"]+" | sed 's/name="confirm" value="//')
file_uuid=$(echo "$confirmation_page" | grep -oE "name=\"uuid\" value=\"[^\"]+" | sed 's/name="uuid" value="//')

# Construct the download URL
download_url="https://drive.usercontent.google.com/download?id=$file_id&export=download&confirm=$file_confirm&uuid=$file_uuid"

# Download the file
curl -L -o "$destination_path" "$download_url"

# Check if the download succeeded
if [ $? -eq 0 ]; then
    echo "Download completed successfully."
else
    echo "Download failed."
fi
