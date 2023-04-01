import os
import subprocess

file_list = os.listdir('../bagfiles/')

for file in file_list:
    if file != '.DS_Store':
        process = subprocess.Popen([
            'python3', 'bagfile_parser.py', os.path.join(str('../bagfiles'),file)
        ])
        print("Processing file:", file)
    
        while True:
            if process.poll() is not None:
                print("Processed finshed:", file)
                break