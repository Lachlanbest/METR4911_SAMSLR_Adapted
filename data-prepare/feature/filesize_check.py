import os

# Specify the directory path
directory_path = '/home/Student/s4582342/wholebody_train'

# Iterate through files in the directory
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)

    # Check if it's a file (not a directory) and has a filesize of 128 bytes
    if os.path.isfile(file_path) and os.path.getsize(file_path) == 128:
        print(f"Filename: {filename}, Filesize: 128 bytes")