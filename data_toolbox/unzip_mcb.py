from zipfile import ZipFile
import glob
from os import path

# Extracts only the needed MCB parts, as the complete dataset is to big to be extracted at once

parts_path = "../../labeled_parts/labels/*.npy"
zip_path = "../../labeled_parts/parts/MCB_Dataset.zip"
extract_folder = "../../labeled_parts/parts"


obj_filepaths = glob.glob(parts_path)
obj_names = [file.split("/")[-1] for file in obj_filepaths]

with ZipFile(zip_path, 'r') as zipObj:
   # Get a list of all archived file names from the zip
   listOfFileNames = zipObj.namelist()
   # Iterate over the file names
   for fileName in listOfFileNames:
       adapted_filename = "XML" + fileName.split("/")[-1].replace('.npy', "_out.npy")   #remove MCB_Dataset infront

       # Check filename endswith csv
       if adapted_filename in obj_names :
           # Extract a single file from zip
           zipObj.extract(fileName, extract_folder)
           print("Extracted " + fileName)