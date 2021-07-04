
from tqdm import tqdm
import os

def get_all_file_paths(directory):

    # initializing empty file paths list
    file_paths = []

    # crawling through directory and subdirectories
    print("get all file paths")
    for root, directories, files in tqdm(os.walk(directory)):
        for filename in files:
            # join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    # returning all file paths
    return file_paths

def create_zipfile(directory):
    # path to folder which needs to be zipped
    #path = '/Users/Salem Rezzag/Desktop/New folder'
    #directory = './images rename'
    from zipfile import ZipFile
    import os
    # calling function to get all file paths in the directory
    file_paths = get_all_file_paths(directory)

    # printing the list of all files to be zipped
    #print('Following files will be zipped in this program:')
    #for file_name in file_paths:
    #    print(file_name)

    # writing files to a zipfile
    print("writing files to a zipfile")
    with ZipFile('myzipfile.zip','w') as zip:
        # writing each file one by one
        for file in tqdm(file_paths):
            zip.write(file)

    print('All files zipped successfully!')