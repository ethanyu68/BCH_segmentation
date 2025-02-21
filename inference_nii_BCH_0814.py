import os, glob

import pandas as pd
import torch
from models_seg_old import DenseTV
#import torchgeometry as tgm
import nibabel as nib
import numpy as np
import argparse
import cv2
import nibabel
'''
By running this script, a folder of raw NIFTI MRI scans will be loaded.
Segmentation of brain into nonbrain/extra-axial-fluid/brain/ventricle/air will be generated.
Two folders (/NIFTI and /PNG) will be created to store the segmentation results.
'''
parser = argparse.ArgumentParser()
parser.add_argument("--block_config", default=((4, 4, 4),(4,4,4)), type=tuple, help='number of dense layers in each Dense Block. Dont change this')
parser.add_argument("--num_classes", default=5, type=int, help='5 classes (labels): 0-nonbrain; 1-EAF;2-brain;3-ventricle;4-air')
parser.add_argument("--combine_csf", default=0, type=int, help='1: combine EAF and ventricle to class 1 as CSF class; 0: output EAF (label 1) and ventricle (label 3)')
parser.add_argument("--path_model",default='/home/ethanyu/MRI_segmentation_pipeline_2024Aug/dense_dmap_EAF_0815/model_epoch_500.pth', help='the model path')
parser.add_argument("--select_mrn", default=0, help='1: if we want to select specific MRN from the folder to segment; 0: segment all the files in the folder.')
parser.add_argument("--select_date", default=0, help='1: if we want to select specific timepoints of MRN from the folder to segment; 0: segment all the files in the folder.')
parser.add_argument("--list_scans", default='/home/ethanyu/data/SB_COHORT.xlsx', help='the xlsx file that contains <MRN> and <mri_date> or just <MRN>')

parser.add_argument("--dir_data_src",default='/home/ethanyu/data/raw_data/SB_0823_nii/', help='The source location that contain folders of subject to be segmented.')
parser.add_argument("--dir_data_dst",default='/home/ethanyu/data/seg_results/SB_0909/', help='The destination location to store the results.')
parser.add_argument("--distance_map", default=1, help='1: add distance map to one of the channels.')
parser.add_argument("--save_nifti", default=1, help='1: save segmentation results in NIFTI format')
parser.add_argument("--save_png", default=0, help='1: save segmentation results in PNG format')
parser.add_argument("--skip_existing_nii", default=0, help='1: if skip the existing NIFTI segmentation files; 0: overwrite existing results.')
parser.add_argument("--skip_existing_png", default=1, help='1: if skip the existing PNG segmentation files; 0: overwrite existing results.')

args = parser.parse_args()
def main():
    # check folder for saving results
    if not os.path.exists(args.dir_data_dst):
        os.makedirs(args.dir_data_dst)
    if not os.path.exists(args.dir_data_dst + '/PNG'):
        os.makedirs(args.dir_data_dst + '/PNG')
    if not os.path.exists(args.dir_data_dst + '/NIFTI'):
        os.makedirs(args.dir_data_dst + '/NIFTI')

    # set up model at this fold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseTV(num_classes=args.num_classes, block_config=args.block_config)
    model = torch.nn.DataParallel(model).to(device)
    state = torch.load(args.path_model)['model']
    model.load_state_dict(state)
    #orientation = np.array([[1, 0, 0, 0, 0, -1], [1,0,0,0,1,0], [0, 1, 0, 0, 0, -1]]) # the second orientation represents axial
    # read data
    if args.select_mrn:
        df = pd.read_excel(args.list_scans)
        ls_target_MRN = set()
        for mrn in df['MRN'].tolist():
            ls_target_MRN.add(mrn)
    # list of subjects
    ls_subj = [x for x in os.listdir(args.dir_data_src) if '.' not in x]
    for subj in ls_subj:
        # if subj != '4983278-THORNTON_AYVELIEN-20160610':
        #     continue 
        if args.select_mrn:
            mrn = int(subj[:7])
            if mrn not in ls_target_MRN:
                continue
            if args.select_date:
                target_dates = df['mri_date'][np.where(df['MRN']==mrn)[0]].tolist() # date format: mm/dd/yyyy
        path_subj = args.dir_data_src + subj
        # list of NIFTI scans of this subject 
        ls_nii = [x for x in os.listdir(path_subj) if 'seg' not in x]
        for f_nii in ls_nii:
            if args.select_date: # if reading csv, check if the scan's date is in the list; otherwise, process all scans.
                scan_date = check_date(f_nii)
                scan_date = str(int(scan_date[4:6])) + '/' + str(int(scan_date[6:8])) + '/' + str(int(scan_date[:4]))
                if scan_date not in target_dates:
                    continue
            path_save = args.dir_data_dst + '/NIFTI/' + subj +'/' + f_nii[:-7] + '_seg.nii.gz'
            # if we choose to skip existing NIFTI segmentation
            if args.skip_existing_nii and os.path.exists(path_save):
                continue
            # if target segmentation folder has not been created, create it.
            if not os.path.exists(args.dir_data_dst + '/NIFTI/' + subj +'/'):
                os.makedirs(args.dir_data_dst + '/NIFTI/' + subj +'/')
            if args.skip_existing_png and os.path.exists(args.dir_data_dst + '/PNG/'+ subj + '_' + f_nii[:-7] + '_15.png'):
                continue
            path_nii = os.path.join(path_subj, f_nii)
            print('Analyzing:', path_nii)
            # if there are slices from 3 planes, remove the sagittal and coronal ones
            # if '3_PLANES' in os.listdir(path_v + '/scans/')[0]:
            #     for path_dcm in glob.glob(path_v +'/*/*/*/*/*/*.dcm'):
            #         dicom = pydicom.read_file(path_dcm)
            #         is_axial = np.where(np.mean(abs(orientation - np.round(dicom.ImageOrientationPatient)), 1) == 0)[0] == 1
            #         if not is_axial:
            #             os.remove(path_dcm)
            # if no NIFTI file is found in the directory, create it by converting DICOM series to NIFTI file     
            # find the file name of .nii.gz file of original scan
            
            nii_orig = nibabel.load(path_nii)
            # standardize the orientation
            nii_orig = nibabel.as_closest_canonical(nii_orig)
            # print('Saving:', path_v + '/' + path_nii)
            # segment the MRI images using the model
            nii_seg = inference_nii(model, nii_orig, args)
            
            if args.save_nifti == 1:
                print('Saving:', path_save)
                nibabel.save(nii_orig, args.dir_data_dst + '/NIFTI/' + subj +'/' + f_nii)
                nibabel.save(nii_seg, path_save)
            
            if args.save_png == 1:
                # # save png
                color_map = {
                    0: (0, 0, 0),    # black
                    1: (255, 0, 0),   # red
                    2: (0, 255, 0),    # green
                    3: (0, 0, 255),  # Blue
                    4: (255, 165, 0)   # Orange
                }
                arr_orig = nii_orig.get_fdata()
                if len(arr_orig.shape) == 4:
                    arr_orig = arr_orig[:,:,:,0]
                arr_seg = nii_seg.get_fdata()
                for i in range(arr_orig.shape[2]):
                    arr_orig_i = arr_orig[:, :, i]/np.max(arr_orig[:, :, i])*255
                    arr_seg_i = arr_seg[:,:,i]
                    arr_orig_i = np.tile(np.rot90(arr_orig_i, k = 1)[:, :, None], (1, 1, 3))
                    arr_seg_i = np.rot90(arr_seg[:,:,i], k = 1)

                    rgb_image = np.zeros((arr_seg_i.shape[0], arr_seg_i.shape[1], 3), dtype=np.uint8)
                    for value, color in color_map.items():
                        rgb_image[arr_seg_i == value] = color

                    orig_seg = np.concatenate([arr_orig_i, rgb_image], 1)
                    cv2.imwrite(args.dir_data_dst + '/PNG/'+ subj + '_' + f_nii[:-7] + '_{}.png'.format(i), orig_seg)
            # # # a = 1


def check_date(s):
    r = len(s)
    l = r - 8
    months = ('01','02','03','04','05','06','07','08','09','10','11','12')
    days = set()
    for i in range(1,32):
        if i < 10:
            days.add('0' + str(i))
        else:
            days.add(str(i))
    while l >= 0:
        if (s[l:l+2] == '20' or s[l:l+2] == '19') and s[l+4:l+6] in months and s[l+6:l+8] in days:
            return s[l:l+8]
        l -= 1
    return ''

def generate_dmap(h,w,rh,rw):
    dmap = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            dmap[i][j] = np.sqrt(((i - h//2)*rh)**2 + ((j - w//2)*rw)**2)
    return np.uint8(dmap)

def process_single(img, dmap=None):
    H, W = img.shape
    # crop the image to make it multiples of 16
    h = H % 16
    hl, hr = h//2, h - h//2
    w = W % 16
    wl, wr = w//2, w - w//2
    img = img[hl:H-hr, wl:W-wr]
    dmap = dmap[hl:H-hr, wl:W-wr]
    img = np.rot90(img, k=1)
    dmap = np.rot90(dmap, k=1)
    if dmap is not None:
        img = np.concatenate([img[None, None, :, :], img[None, None, :, :], dmap[None, None, :, :]], 1)
    else:
        img = np.concatenate([img[None, None, :, :], img[None, None, :, :], img[None, None, :, :]], 1)
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])
    img = (img - mean) / std
    return img


def normalize_intensity(array,  margin_ratio=20, vmax_percent=99.7, vmin_percent =90):
    '''
        this function normalizes the intensity stack-wise.
        vmax is chosen as the 99.7% percent of the non-zero intensities of the stack
        vmin is chosen as the 90% percent of the margin/boundaries of the whole stack. The margin is controlled by the margin_ratio.
    '''
    for i in range(array.shape[2]):
        img = array[:, :, i]
        nonzero = img > 0
        if np.sum(nonzero) == 0:
            continue
        vmax = np.percentile(img[nonzero], vmax_percent)
        img[img > vmax] = vmax
        img = img/(vmax + 0.000001)
        array[:, :, i] = img
    return array


def inference_nii(model, nii, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    data = nii.get_fdata()
    if len(data.shape) == 4:
        data = data[:,:,:,0]
    H, W, num_img = data.shape
    # intensity normalization
    data = normalize_intensity(data)
    res = nii.header.get_zooms()
    dmap = generate_dmap(H, W, res[0], res[1])/255
    #data = check_orientation(nii, data)
    seg_results = []
    batch = []
    for j in range(num_img):
        img = data[:, :, j]
        # the head is towards left
        img = process_single(img, dmap)
        batch.append(img)
        if len(batch) < 16 and j < num_img-1:
            continue
        batch = np.concatenate(batch, 0)
        batch = torch.from_numpy(batch).float().to(device)
        with torch.no_grad():
            out, _, _, _ = model(batch)
            #out = model(batch)
        out = out.cpu().numpy()
        for i in range(out.shape[0]):
            orig_size = np.zeros([H, W])
            pred = np.argmax(out[i], 0).astype(np.float32) # shape: B x H x W, values of each pixel: 0 - 5
            pred = np.rot90(pred, k=-1)
            orig_size[H//2 - pred.shape[0]//2:H//2 + pred.shape[0]//2, W//2 - pred.shape[1]//2 : W//2 + pred.shape[1]//2] = pred
            seg_results.append(orig_size)
        batch = []

    seg_results = np.array(seg_results)
    seg_results = np.round(np.rollaxis(seg_results, 0, 3))
    if args.combine_csf == 1:
        seg_results[seg_results == 3] = 1
    # To avoid the ITK error, the air is combined to the non-brain class.
    if np.count_nonzero(seg_results == 4) < 10:
        seg_results[seg_results == 4] = 0

    seg_result_nii = nib.Nifti1Image(seg_results, nii.affine, nii.header)
    return seg_result_nii



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
