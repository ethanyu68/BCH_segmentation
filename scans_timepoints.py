import os
import pandas as pd
import numpy as np
from datetime import date
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--src",default='/home/ethanyu/data/raw_data/BV_CSFV_nii/', help='folder that stores all the NIFTI files')
parser.add_argument("--dst",default='/home/ethanyu/data/raw_data/BV_CSFV_pullETV_Info.xlsx', help='save spreasheet file to this path ')
parser.add_argument("--info",default='/home/ethanyu/data/BV_CSFV_STUDY.xlsx', help='the spreadsheet that has all the information of the patients.')
parser.add_argument("--info_sheet",default='etv_data_all', help='the spreadsheet that has all the information of the patients.')
parser.add_argument("--mrn_ls",default='/home/ethanyu/data/BV_CSFV_STUDY.xlsx', help='the xlsx file that contains <MRN> column')
parser.add_argument("--mrn_ls_sheet",default='etv_data_all', help='the xlsx file that contains <MRN> column')

parser.add_argument("--col_mrn", default='MRN', help='the column in info sheet about MRN')
parser.add_argument("--col_dob", default='DOB_corr', help='the column in info sheet about patient date of birth')
parser.add_argument("--col_date_sx", default='date_sx', help='the column in info sheet about date of surgery')
parser.add_argument("--col_gender", default='Gender', help='the column in info sheet about patient gender')


args = parser.parse_args()

def main():
    BV_CSFV_STUDY = pd.read_excel(args.info, sheet_name=args.info_sheet)
    mrn_ls = pd.read_excel(args.mrn_ls, sheet_name=args.mrn_ls_sheet)['MRN']
    src = args.src
    dst = args.dst
    missed = []
    df = {'MRN':[], 'date_surgery': [], 'mri_date': [], 'days_postop':[], 'dob_corr':[], 'age':[], 'gender':[]}
    for i in range(len(mrn_ls)):
        mrn = mrn_ls[i]
        fdr = [x for x in os.listdir(src) if str(mrn) in x]
        if len(fdr) == 0:
            missed.append(mrn)
            continue
        else:
            fdr = fdr[0]
        ls_scans = os.listdir(src + fdr)
        for scan in ls_scans:
            df['MRN'].append(mrn)
            date_scan = check_date(scan)
            if '_' in date_scan[:4]:
                print(scan)
            date_scan = date_scan[4:6] + '/' + date_scan[6:8] + '/' + date_scan[:4]
            
            df['mri_date'].append(date_scan)
            if len(np.where(BV_CSFV_STUDY[args.col_mrn] == int(mrn))[0]) == 0:
                continue
            # baseline = BV_CSFV_STUDY[args.col_baseline][np.where(BV_CSFV_STUDY[args.col_mrn] == int(mrn))[0].item()]
            # df['baseline'].append(baseline)
            # preop = BV_CSFV_STUDY[args.col_preop][np.where(BV_CSFV_STUDY[args.col_mrn] == int(mrn))[0].item()]
            
            # df['preop'].append(preop)
            dob = BV_CSFV_STUDY[args.col_dob][np.where(BV_CSFV_STUDY[args.col_mrn] == int(mrn))[0][0].item()]
            df['dob_corr'].append(dob)
            #print(mrn, dob_date, date, baseline_date, preop_date)
            if 'nan' not in date_scan:
                date_scan_m, date_scan_d, date_scan_y = date_scan.split('/')
                date_scan_days = date(int(date_scan_y), int(date_scan_m), int(date_scan_d))
                dob_days = date(dob.year, dob.month, dob.day)
                df['age'].append(date_scan_days - dob_days)
            else:
                df['age'].append('nan')
            date_surgery = BV_CSFV_STUDY[args.col_date_sx][np.where(BV_CSFV_STUDY[args.col_mrn] == int(mrn))[0][0].item()]
            df['date_surgery'].append(date_surgery)
            date_surgery_days = date(date_surgery.year, date_surgery.month, date_surgery.day)
            df['days_postop'].append(date_scan_days - date_surgery_days)
            # gender
            gender = BV_CSFV_STUDY[args.col_gender][np.where(BV_CSFV_STUDY[args.col_mrn] == int(mrn))[0][0].item()]
            if gender == 1:
                df['gender'].append('Male')
            else:
                df['gender'].append('Female')
    print("The missing MRNs: {}".format(missed))

    df = pd.DataFrame(df)
    df.to_excel(dst)



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
        if '_' not in s[l:l+8] and '-' not in s[l:l+8] and (s[l:l+2] == '20' or s[l:l+2] == '19') and s[l+4:l+6] in months and s[l+6:l+8] in days:
            return s[l:l+8]
        l -= 1
    return ''

if __name__ == '__main__':
    main()