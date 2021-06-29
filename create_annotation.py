
import os
import pandas as pd
import argparse


def create_annotation(path, a):

   
    images_path = os.path.join(path,'Images')
    #masks_path = os.path.join(path,'Ground-truths')
    
    images = os.listdir(images_path)
    #masks = os.listdir(masks_path)

    covid_images =[image for image in images]
    #no_covid_images =[image for image in images if 'mask_'+image not in masks]

    covid = pd.DataFrame(columns=['img','target'])
    #no_covid = pd.DataFrame(columns=['img','target'])

    covid['img'] = covid_images
    covid['target'] = a
    #no_covid['img'] = no_covid_images
    #no_covid['target'] = 0

    annotation = pd.concat([covid])

    annotation = annotation.reset_index()

    return annotation

def get_args():

    parser = argparse.ArgumentParser(description = "Qata_Covid19 Segmentation" ,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set your environment
    parser.add_argument('--path',type=str,default='./data/Qata_COV')
    parser.add_argument('--a', type=str, default = '0')
    # arguments for training
    #parser.add_argument('--img_size', type = int , default = 224)

    #parser.add_argument('--load_model', type=str, default='best_checkpoint.pt', help='.pth file path to load model')

    parser.add_argument('--out', type=str, default='./dataset')
    return parser.parse_args()

def main():
    
    args = get_args()

    if not os.path.exists(args.out):
        print("path created")
        os.mkdir(args.out)
    
    df = create_annotation(args.path, args.a)

    df.to_csv(os.path.join(args.out,'target.csv'),index=False)