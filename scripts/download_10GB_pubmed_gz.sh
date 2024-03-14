#!/bin/bash

MAX_SIZE=$((10 * 1024 * 1024 * 1024))  # 10 GB in bytes
current_size=0

while IFS= read -r file; do
    # Check the size of the next file
    file_size=$(wget --spider --server-response "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/${file}" 2>&1 | grep -oP '(?<=Content-Length: )\d+' | head -n 1)
    new_size=$((current_size + file_size))

    # Check if adding the next file exceeds the limit
    if (( new_size > MAX_SIZE )); then
        echo "Reached approximately 10 GB total size, stopping."
        break
    else
        # Download the file since it doesn't exceed the limit
        wget "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/${file}"
        current_size=$new_size
    fi
done < file_list.txt
