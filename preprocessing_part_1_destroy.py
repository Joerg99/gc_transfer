# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 25/03/2020
"""
First preprocessing program that :
- generates Chargrids from input images thanks to Tesseract
- extracts bounding boxes for each class from the ground truth files
- generates class segmentation from the class bounding boxes
- reduces the size of images by removing empty rows and empty columns

Requirements
----------
- Tesseract must be installed in "C:\Program Files\Tesseract-OCR/tesseract"
- Input images must be located in the folder dir_img = "./data/img_inputs/"
- Input bounding boxes (ground truth) must be located in the folder dir_boxes = "./data/gt_boxes/"
- Input classes (ground truth) must be located in the folder dir_classes = "./data/gt_classes/"

Hyperparameters
----------
- tesseract_conf_threshold : gives a threshold below which the tesseract information is not kept
- cosine_similarity_threshold : gives a threshold above which two strings are considered similar

Return
----------
Several files are generated :
- in outdir_np_chargrid = "./data/np_chargrids/" : Chargrids of each input image in npy (numpy array format)
- in outdir_png_chargrid = "./data/img_chargrids/" : Chargrids of each input image in png
- in outdir_np_gt = "./data/np_gt/" : Class Segmentation of each input image in npy (numpy array format)
- in outdir_png_gt = "./data/img_gt/" : Class Segmentation of each input image in png
- in outdir_pd_bbox = "./data/pd_bbox/" : Class Bounding Boxes of each input image in pkl (pandas dataframe format)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract as te
import os
import re
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

te.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR/tesseract'

## Hyperparameters
dir_img = "./data/img_inputs/"
dir_boxes = "./data/gt_boxes/"
dir_classes = "./data/gt_classes/"
outdir_np_chargrid = "./data/np_chargrids/"
outdir_png_chargrid = "./data/img_chargrids/"
outdir_np_gt = "./data/np_gt/"
outdir_png_gt = "./data/img_gt/"
outdir_pd_bbox = "./data/pd_bbox/"
tesseract_conf_threshold = 10
cosine_similarity_threshold = 0.4
list_classes = ["company", "date", "address", "total"]
nb_classes = len(list_classes)

def add_row_gt_pd(row, c, gt_pd):
    return gt_pd.append({
            'left':row['top_left_x'],
            'top':row['top_left_y'],
            'right':row['bot_right_x'],
            'bot':row['bot_right_y'],
            'class':c
            }, ignore_index = True)

def extract_tesseract_information(filename):
    img = plt.imread(os.path.join(dir_img, filename), format='jpeg')
    print(filename, img.shape)
    
    tesseract_out = te.image_to_data(img, config="", output_type=te.Output.DATAFRAME)
    tesseract_out = tesseract_out[tesseract_out['conf']>tesseract_conf_threshold]
    tesseract_out["text"] = tesseract_out["text"].astype('str')
    
    return tesseract_out, img.shape

def get_chargrid(tesseract_out):
    chargrid_pd = pd.DataFrame(columns = ['left', 'top', 'width', 'height', 'ord', 'conf'])

    for index, row in tesseract_out.iterrows():
        for i in range(0, len(row["text"])):
            row['width'] = (row['width']+len(row["text"])-1)//len(row["text"])*len(row["text"]) # Split character by character
        
            chargrid_pd = chargrid_pd.append({
                'left':row['left']+row['width']*i//len(row["text"]),
                'top':row['top'],
                'width':row['width']//len(row["text"]),
                'height':row['height'],
                'ord':ord(row["text"][i]),
                'conf':row["conf"]
            }, ignore_index = True)
    # ord >= 33 and <= 126 non readable characters
    # -32 to start at 1. Ok, because it's just a unique encoding of chars.
    chargrid_pd = chargrid_pd[chargrid_pd['ord']>=33]
    chargrid_pd = chargrid_pd[chargrid_pd['ord']<=126]
    chargrid_pd['ord'] -= 32
    
    return chargrid_pd
    
def extract_class_bounding_boxes(filename):
    '''
        classes: 1 = total, 2 = address , 3 = company ,4 = date

    :param filename:
    :return: gt_pd which is a pandas dataframe that has the bounding boxes considered as labels
            (dataset has bounding box for every word, so bbox labels have to be constructed first)
    '''
    gt_pd = pd.DataFrame(columns = ['left', 'top', 'right', 'bot', 'class'])
    
    ## Import ground truth files
    pd_boxes = pd.DataFrame(columns=['top_left_x', 'top_left_y', 'top_right_x', 'top_right_y', 'bot_left_x', 'bot_left_y', 'bot_right_x', 'bot_right_y', 'text'])
    dic_class = dict()
    
    with open(os.path.join(dir_boxes, filename).replace("jpg", "txt")) as f:
        reader = f.read().splitlines()
        pd_boxes = pd.DataFrame([x.split(",", 8) for x in reader], columns=['top_left_x', 'top_left_y', 'top_right_x', 'top_right_y', 'bot_right_x', 'bot_right_y', 'bot_left_x', 'bot_left_y', 'text'])
    
        pd_boxes["top_left_x"] = pd_boxes["top_left_x"].astype('int')
        pd_boxes["top_left_y"] = pd_boxes["top_left_y"].astype('int')
        pd_boxes["top_right_x"] = pd_boxes["top_right_x"].astype('int')
        pd_boxes["top_right_y"] = pd_boxes["top_right_y"].astype('int')
        pd_boxes["bot_left_x"] = pd_boxes["bot_left_x"].astype('int')
        pd_boxes["bot_left_y"] = pd_boxes["bot_left_y"].astype('int')
        pd_boxes["bot_right_x"] = pd_boxes["bot_right_x"].astype('int')
        pd_boxes["bot_right_y"] = pd_boxes["bot_right_y"].astype('int')
        pd_boxes["text"] = pd_boxes["text"].str.upper()
    
    with open(os.path.join(dir_classes, filename).replace("jpg", "txt")) as f:
        dic_class = json.load(f)
    for i in range(nb_classes):
        if list_classes[i] not in dic_class.keys():
            dic_class[list_classes[i]] = "UNKNOWN"
        dic_class[list_classes[i]] = dic_class[list_classes[i]].upper()
    print("dic_class", dic_class)
    ## Detect classes in the bounding box file
    vectorized_text = CountVectorizer().fit_transform([dic_class[list_classes[i]] for i in range(nb_classes)]+pd_boxes["text"].tolist()) # countvectorize labels(dic_ and all items in pd_box

    for index, row in pd_boxes.iterrows():
        #Classes of type string

        # vectorized_text[0] = company vs. every row after labels (which is vectorized_text[index+nb_classes]) company --> class 3
        if cosine_similarity(vectorized_text[0].reshape(1, -1), vectorized_text[index+nb_classes].reshape(1, -1))[0][0] > cosine_similarity_threshold:
            gt_pd = add_row_gt_pd(row, 3, gt_pd)

        # vectorized_text[2] = address --> class 2
        if cosine_similarity(vectorized_text[2].reshape(1, -1), vectorized_text[index+nb_classes].reshape(1, -1))[0][0] > cosine_similarity_threshold:
            gt_pd = add_row_gt_pd(row, 2, gt_pd)
        
        #Classes of type date --> class 4
        tab_date = re.findall(r'((?i)(?:[12][0-9]|3[01]|0*[1-9])(?P<sep>[- \/.\\])(?P=sep)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P=sep)+(?:19|20)\d\d|(?:[12][0-9]|3[01]|0*[1-9])(?P<sep2>[- \/.\\])(?P=sep2)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P=sep2)+\d\d|(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P<sep3>[- \/.\\])(?P=sep3)*(?:[12][0-9]|3[01]|0*[1-9])(?P=sep3)+(?:19|20)\d\d|(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P<sep4>[- \/.\\])(?P=sep4)*(?:[12][0-9]|3[01]|0*[1-9])(?P=sep4)+\d\d|(?:19|20)\d\d(?P<sep5>[- \/.\\])(?P=sep5)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P=sep5)+(?:[12][0-9]|3[01]|0*[1-9])|\d\d(?P<sep6>[- \/.\\])(?P=sep6)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P=sep6)+(?:[12][0-9]|3[01]|0*[1-9])|(?:[12][0-9]|3[01]|0*[1-9])(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:19|20)\d\d|(?:[12][0-9]|3[01]|0*[1-9])(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\d\d|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])(?:19|20)\d\d|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])\d\d|(?:19|20)\d\d(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])|\d\d(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])|(?:[12][0-9]|3[01]|0[1-9])(?:1[012]|0[1-9])(?:19|20)\d\d|(?:1[012]|0[1-9])(?:[12][0-9]|3[01]|0[1-9])(?:19|20)\d\d|(?:19|20)\d\d(?:1[012]|0[1-9])(?:[12][0-9]|3[01]|0[1-9])|(?:1[012]|0[1-9])(?:[12][0-9]|3[01]|0[1-9])\d\d|(?:[12][0-9]|3[01]|0[1-9])(?:1[012]|0[1-9])\d\d|\d\d(?:1[012]|0[1-9])(?:[12][0-9]|3[01]|0[1-9]))', row["text"])
        for dat in tab_date:
            if dat[0] == dic_class["date"]:
                gt_pd = add_row_gt_pd(row, 4, gt_pd)
        
        #Classes of type float --> total amount class 1
        tab_floats = re.findall(r'([-+]?[0-9]*\.?[0-9]+)', row["text"])
        total_float = re.search(r'([-+]?[0-9]*\.?[0-9]+)', dic_class["total"])
        if total_float:
            for flo in tab_floats:
                if float(total_float.group(0)) == float(flo):
                    gt_pd = add_row_gt_pd(row, 1, gt_pd)
    
    return gt_pd

def plot_input_vs_output(input, output):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(input)
    ax2.imshow(output)
    plt.show()
    plt.clf()

def get_reduced_output(chargrid_pd, gt_pd, img_shape):
    chargrid_np = np.array([0]*img_shape[0]*img_shape[1]).reshape((img_shape[0], img_shape[1]))
    
    chargrid_pd.sort_values(by="conf", ascending=True, inplace=True) #Sort by confidence
    chargrid_pd.reset_index(drop=True, inplace=True)
    
    for index, row in chargrid_pd.iterrows():
        chargrid_np[row['top']:row['top']+row['height'], row['left']:row['left']+row['width']] = row['ord']
    
    gt_np = np.array([0]*img_shape[0]*img_shape[1]).reshape((img_shape[0], img_shape[1]))
    
    gt_pd.sort_values(by="class", ascending=True, inplace=True) #Sort by confidence
    gt_pd.reset_index(drop=True, inplace=True)
    
    for index, row in gt_pd.iterrows():
        gt_np[row['top']:row['bot'], row['left']:row['right']] = row['class']
    
    ## Remove empty rows and columns
    tab_cumsum_todelete_x = np.cumsum(np.all(chargrid_np == 0, axis=0))
    gt_pd['left'] -= tab_cumsum_todelete_x[gt_pd['left'].tolist()]
    gt_pd['right'] -= tab_cumsum_todelete_x[gt_pd['right'].tolist()]
    
    tab_cumsum_todelete_y = np.cumsum(np.all(chargrid_np == 0, axis=1))
    gt_pd['top'] -= tab_cumsum_todelete_y[gt_pd['top'].tolist()]
    gt_pd['bot'] -= tab_cumsum_todelete_y[gt_pd['bot'].tolist()]
    
    gt_np = gt_np[:,~np.all(chargrid_np == 0, axis=0)]
    gt_np = gt_np[~np.all(chargrid_np == 0, axis=1),:]
    
    chargrid_np = chargrid_np[:,~np.all(chargrid_np == 0, axis=0)]
    chargrid_np = chargrid_np[~np.all(chargrid_np == 0, axis=1),:]
    
    return chargrid_np, gt_np, gt_pd

def create_rectangle_and_correct():
    '''
    work on unscaled np data for input chargrid and label
    labels are too big and include background pixel -> cut off with mask.
    this code doesn't scale the data
    :return:
    '''
    cg_mask = chargrid_np != 0
    label_new = cg_mask * gt_np
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(chargrid_np)
    ax1.add_patch(patches.Rectangle((93, 161), 352 - 93, 183 - 161, linewidth=1, edgecolor='r', facecolor='none'))
    ax1.add_patch(patches.Rectangle((49, 201), 391 - 49, 227 - 201, linewidth=1, edgecolor='r', facecolor='none'))
    ax2.imshow(gt_np)
    ax3.imshow(label_new)
    plt.show()

if __name__ == "__main__":
    list_filenames = [f for f in os.listdir(dir_img) if os.path.isfile(os.path.join(dir_img, f)) and os.path.isfile(os.path.join(dir_boxes, f).replace("jpg", "txt")) and os.path.isfile(os.path.join(dir_classes, f).replace("jpg", "txt"))]

    # c = np.load("./data/np_chargrid_unscaled_top/X00016469612.npy")
    # plt.imshow(c[::2, 1::2])
    # l = np.load("./data/np_label_unscaled_top/X00016469612.npy")
    # plt.imshow(l[::2, 1::2])
    # fix, (ax1, ax2) = plt.subplots(1,2)
    # ax1.imshow(c[::2, 1::2])
    # ax2.imshow(l[::2, 1::2])


    print("Number of input files : ", len(list_filenames))
     
    for filename in list_filenames:

        tesseract_out, img_shape = extract_tesseract_information(filename)
        chargrid_pd = get_chargrid(tesseract_out)

        # dataset label is a dictionary (company: blah blah, date: ...)
        # dataset "gt_boxes" is a bounding box around every element in the data
        # label value may correspond to multiple bounding boxes in gt_boxes. merging of bboxes with CountVect, regex etc.
        # classes: 1 = total, 2 = address , 3 = company ,4 = date
        # not needed with our data!!!
        bbox_label_df = extract_class_bounding_boxes(filename) #rubbish!!!!!!!! but better than nothing

        chargrid_np = np.array([0] * img_shape[0] * img_shape[1]).reshape((img_shape[0], img_shape[1]))
        chargrid_pd.sort_values(by="conf", ascending=True, inplace=True)  # Sort by confidence
        chargrid_pd.reset_index(drop=True, inplace=True)
        for index, row in chargrid_pd.iterrows():
            chargrid_np[row['top']:row['top'] + row['height'], row['left']:row['left'] + row['width']] = row['ord']
        gt_np = np.array([0] * img_shape[0] * img_shape[1]).reshape((img_shape[0], img_shape[1]))
        bbox_label_df.sort_values(by="class", ascending=True, inplace=True)  # Sort by confidence
        bbox_label_df.reset_index(drop=True, inplace=True)
        for index, row in bbox_label_df.iterrows():
            gt_np[row['top']:row['bot'], row['left']:row['right']] = row['class']
        import matplotlib.patches as patches
        #chargrid_np, gt_np, bbox_label_df = get_reduced_output(chargrid_pd, bbox_label_df, img_shape)

        outdir_np_cg_unscaled_top = "./data/np_chargrid_unscaled_top/"
        outdir_np_label_unscaled_top = "./data/np_label_unscaled_top/"
        chargrid_np = chargrid_np[:384, :384]
        gt_np = gt_np[:384, :384]
        chargrid_np = chargrid_np[::2, 1::2]
        gt_np = gt_np[::2, 1::2]
        cg_mask = chargrid_np != 0
        label_new = cg_mask * gt_np
        np.save(os.path.join(outdir_np_cg_unscaled_top, filename).replace("jpg", "npy"), chargrid_np)
        np.save(os.path.join(outdir_np_label_unscaled_top, filename).replace("jpg", "npy"), label_new)

        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # ax1.imshow(chargrid_np)
        # ax1.add_patch(patches.Rectangle((93, 161), 352-93, 183-161, linewidth=1, edgecolor='r', facecolor='none'))
        # ax1.add_patch(patches.Rectangle((49, 201), 391 - 49, 227 - 201, linewidth=1, edgecolor='r', facecolor='none'))
        # ax2.imshow(gt_np)
        # ax3.imshow(label_new)
        # plt.show()
    #plot_input_vs_output(chargrid_np, gt_np)
    #print(gt_pd)

    print('stop')

        ##Saving
        # np.save(os.path.join(outdir_np_chargrid, filename).replace("jpg", "npy"), chargrid_np)
        # np.save(os.path.join(outdir_np_gt, filename).replace("jpg", "npy"), gt_np)
        # bbox_label_df.to_pickle(os.path.join(outdir_pd_bbox, filename).replace("jpg", "pkl"))
        #
        # plt.imshow(chargrid_np)
        # plt.savefig(os.path.join(outdir_png_chargrid, filename).replace("jpg", "png"))
        # plt.close()
        #
        # plt.imshow(gt_np)
        # plt.savefig(os.path.join(outdir_png_gt, filename).replace("jpg", "png"))
        # plt.close()
