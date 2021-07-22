
from tqdm import tqdm
from csv import reader
import argparse
import os
import pandas as pd

def get_args():

    parser = argparse.ArgumentParser(description = "U-Net for Lung Segmentation" ,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set your environment
    parser.add_argument('--path_pneumonia_csv', type=str, default = 'E:/2 MASTER/Memoire/07-06-2021 (croped)/croped pneumonia/zipf csv')
    parser.add_argument('--path_covid_csv', type=str, default = 'E:/2 MASTER/Memoire/07-06-2021 (croped)/covid_croped/zipf csv')
    parser.add_argument('--path_normal_csv', type=str, default = 'E:/2 MASTER/Memoire/07-06-2021 (croped)/normal_croped/zipf csv')
    parser.add_argument('--out', type=str, default = 'E:/2 MASTER/Memoire/07-06-2021 (croped)')
    return parser.parse_args()

args = get_args()

#df = pd.read_csv(os.path.join(args.path,'target.csv'))
Data = []
#Data = pd.DataFrame(columns=['index','img','target'])
# open file in read mode
#with open(args.path_pneumonia_csv + '/data.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
#    csv_reader = reader(read_obj)
#    header = next(csv_reader)
    # Iterate over each row in the csv using reader object
#    if header != None:
#        for row in tqdm(csv_reader):
        # row variable is a list that represents a row in csv
        #Data.append([row], ignore_index=True)
        #Data['img'] = row['img']
        #Data['target'] = row['target']
#            Data.append(row)
        #print(row)

with open(args.path_covid_csv + '/data.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    # Iterate over each row in the csv using reader object
    if header != None:
        for row in tqdm(csv_reader):
        # row variable is a list that represents a row in csv
        #Data.append([row], ignore_index=True)
        #Data['img'] = row['img']
        #Data['target'] = row['target']
            Data.append(row)

with open(args.path_normal_csv + '/data.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    # Iterate over each row in the csv using reader object
    if header != None:
        for row in tqdm(csv_reader):
        # row variable is a list that represents a row in csv
        #Data.append([row], ignore_index=True)
        #Data['img'] = row['img']
        #Data['target'] = row['target']
            Data.append(row)

feature_df = pd.DataFrame(Data)
#feature_df.rename(columns=({ '0': 'index', '1': 'img', '2': 'target'}), inplace=True,)

feature_df = feature_df.drop(0, axis=1)
feature_df = feature_df.reset_index()
feature_df.columns = ['index', 'img', 'target', '0', '1', '2', '3', '4', '5', '6', '7']

feature_df.to_csv(os.path.join(args.out,'data_concat_non_normaliz_2_class.csv'),index=False)