import os
import subprocess

file_list = os.listdir('../bagfiles/')

print("Number of files:", len(file_list))

for idx, file in enumerate(file_list):    
    print("\n\n\nProcessing Index:", idx)    
    
    if file != '.DS_Store':
        process = subprocess.Popen([
            'python3', 'bagfile_parser.py', os.path.join(str('../bagfiles'),file)
        ])
        print(">>>>>>>>>>>>>>>>>> Processing file:", file)
    
        while True:
            if process.poll() is not None:
                print("\n\n\n >>>>>>>>>>>>>>>>>> Processed finshed:", file)
                break        
    else:
        print('Skipping file:', file)