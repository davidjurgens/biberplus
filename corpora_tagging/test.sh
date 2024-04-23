#!/bin/bash

# Path to the Jupyter notebook (properly quoted)
notebook_path="tag_reddit.ipynb"
log_file="notebook_processing.log"

# Generate an array of filenames
filenames=()
for i in {11..1}; do
    filenames+=("RC_2019-$(printf "%02d" $i).gz")
done

# Initialize or clear the log file
echo "Starting the notebook processing batch." > $log_file

# Loop through each filename and update the notebook
for new_filename in "${filenames[@]}"; do
    echo "$(date +"%Y-%m-%d %H:%M:%S") - Starting to process $new_filename" >> $log_file

    # Use Python to safely edit the JSON content of the notebook
    python -c "
import json

# Open and load the notebook
with open('$notebook_path', 'r', encoding='utf-8') as file:
    notebook = json.load(file)

# Edit the notebook content
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        cell['source'] = [line if 'file_name =' not in line else \"file_name = '$new_filename'\" for line in cell['source']]

# Save the changes back to the file
with open('$notebook_path', 'w', encoding='utf-8') as file:
    json.dump(notebook, file, indent=2)
    "
    echo "$(date +"%Y-%m-%d %H:%M:%S") - Finished updating notebook for $new_filename" >> $log_file

    # Execute the notebook using Jupyter nbconvert and save each output uniquely
    output_path="${notebook_path%.ipynb}_$new_filename.ipynb"
    echo "$(date +"%Y-%m-%d %H:%M:%S") - Starting execution of $new_filename" >> $log_file
    jupyter nbconvert --to notebook --execute "$notebook_path" --output "$output_path"
    echo "$(date +"%Y-%m-%d %H:%M:%S") - Completed execution of $new_filename" >> $log_file

    # Log completion
    echo "$(date +"%Y-%m-%d %H:%M:%S") - Processed and executed notebook with $new_filename" >> $log_file

    # Wait or perform other actions if necessary
    sleep 1  # Delay for demonstration; adjust as needed
done

echo "$(date +"%Y-%m-%d %H:%M:%S") - Completed all notebook processing." >> $log_file
