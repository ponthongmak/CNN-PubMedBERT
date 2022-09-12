# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 13:30:36 2020

@author: Wanchana
"""
import pandas as pd
import os
import numpy as np
from datetime import timedelta
from matplotlib import pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import MultiLabelBinarizer

# path
path = ''
os.chdir(path)

#### import data
# =============================================================================
# demographic data
# =============================================================================
df_demo = pd.read_excel('')
	# check duplicate
df_demo = df_demo.drop_duplicates()
	# convert date
df_demo['DOB'] = pd.to_datetime(df_demo['DOB'], errors='coerce')
	# convert sex
df_demo.SEX.replace({'M': 1, 'F': 2}, inplace=True)
# =============================================================================
# admission data
# =============================================================================
df_admit = pd.read_excel()
	# check duplicate
df_admit = df_admit.drop_duplicates()
	# covert date
df_admit['DC_DATE'] = pd.to_datetime(df_admit['DC_DATE'], errors='coerce')
df_admit = df_admit[['HN', 'AN', 'DC_DATE', 'AWARD', 'DWARD']]

	# check start and end date
print(df_admit.DC_DATE.dropna().min())
print(df_admit.DC_DATE.dropna().max())
	# 2010-01-01 00:00:00
	# 2019-12-31 00:00:00
# =============================================================================
# discharge summary data
# =============================================================================
df_dcs = pd.read_excel('')
	# check duplicate
df_dcs = df_dcs.drop_duplicates()
	# covert date
df_dcs['KEYDATE'] = pd.to_datetime(df_dcs['KEYDATE'], errors='coerce')

df_dcs = df_dcs[['HN', 'AN', 'KEYDATE', 'DPT', 'BRIEF', 'COURSE']]
df_dcs = df_dcs.sort_values(['HN', 'KEYDATE'], ascending=[True, True])

	# check start and end date
print(df_dcs.KEYDATE.dropna().min())
print(df_dcs.KEYDATE.dropna().max())
	# 2005-05-02 00:00:00
	# 2020-08-13 00:00:00
# =============================================================================
# diagnosis data
# =============================================================================
df_diag = pd.read_csv('')
	# check duplicate
df_diag = df_diag.drop_duplicates()
df_diag = df_diag.drop_duplicates(subset=['AN', 'DX_CODE'], keep='first')
	# covert date
df_diag['DX_SEQ'] = pd.to_numeric(df_diag['DX_SEQ'])
df_diag['DX_DATE'] = pd.to_datetime(df_diag['DX_DATE'], errors='coerce')
df_diag = df_diag.sort_values(['HN', 'AN', 'DX_DATE', 'DX_SEQ'], ascending=[
                              True, True, True, True])
# =============================================================================
# ## ICD data
# =============================================================================
df_ICD = pd.read_excel('')
df_ICD = df_ICD.drop_duplicates()
# =============================================================================
# merge data
# =============================================================================
## merge patient demo
df1 = df_admit.merge(df_demo, how='left', on='HN')

## merge diagnosis
df2 = df1.merge(df_diag, how='left', on=['HN', 'AN'])

## merge text
df3 = df2.merge(df_discharge, how='left', on=['HN', 'AN'])

# merge ICD-10
df3 = df3.merge(df_ICD, how='left', left_on='DX_CODE', right_on='ICD10')

# cap start and end date
start_date = '01-01-2015 00:00:00'
end_date = '31-12-2019 00:00:00'
mask = (df3['DC_DATE'] > start_date) & (df3['DC_DATE'] <= end_date)
df3 = df3.loc[mask]

print(df3.DC_DATE.dropna().min())
print(df3.DC_DATE.dropna().max())
# =============================================================================
# select sample by inclusion criteria
# =============================================================================
df3[['DX_CODE', 'BRIEF', 'COURSE']] = df3[['DX_CODE', 'BRIEF', 'COURSE']].replace('-', np.nan)
    # n of missing records
df4 = df3[~(df3.DX_CODE.isna() | df3.BRIEF.isna() | df3.COURSE.isna())]

# check year
df4 = df4[['HN', 'AN', 'SEX', 'DOB', 'DC_DATE', 'BRIEF',
           'COURSE', 'DX_SEQ', 'DX_CODE', 'ICD10_des']]
df4 = df4.rename(columns={'HN': 'hn', 'AN': 'an', 'SEX': 'gender',
                          'DC_DATE': 'date', 'BRIEF': 'brief', 
				  'COURSE': 'course', 'DX_CODE': 'dx'})
# calcuate age
df4['age'] = (df4.date - df4.DOB) // timedelta(days=365.2425)

df4.an.nunique()
# 21263
df4.hn.nunique()
# 12194

## cap age >=18
df4 = df4[df4.age >=18]
df4.an.nunique()
# 21130
df4.hn.nunique()
# 12146
# =============================================================================
# create group of ICD by ICD chapter
# =============================================================================
df4 = df4.sort_values(['an', 'DX_SEQ'], ascending=[True, True])

## check multiple pdx cases
df_a1 = df4[df4.DX_SEQ == 1]
df_a2 = df_a1[df_a1.duplicated('an', keep = False)]
df_a2 = df_a2[['hn','an','biref','course','DX_SEQ','dx','ICD10_des']]

df_a2.an.nunique()
    # 539 admissions have more than 1 pdx

## no pdx case
df4['count'] = df4.groupby(['an'])['DX_SEQ'].transform(lambda x: sum(x == 1))
x = df4[df4['count'] == 0]
df4 =df4.drop(columns = {'count'})
df_a3 = x[['hn','an','biref','course','DX_SEQ','dx','ICD10_des']]

df_a3.an.nunique()
    # 41 admissions have no than 1 pdx
del df_a1, df_a2, df_a3
# =============================================================================
# all these are required to verified by medical doctor team
x.to_excel('review_no_pdx.xlsx')
df_a2.to_excel('review_more_than_one_pdx.xlsx')
# =============================================================================
    # load verified df no pdx
df_no_pdx_update = pd.read_excel('review_no_pdx.xlsx', dtype=str)
df_no_pdx_update['DX_SEQ'] = pd.to_numeric(df_no_pdx_update['DX_SEQ'])
df_no_pdx_update['SEQ_NEW'] = pd.to_numeric(df_no_pdx_update['SEQ_NEW'])

df4 = df4.merge(df_no_pdx_update[['an', 'DX_SEQ', 'SEQ_NEW']], 
		    how='left', on=['an', 'DX_SEQ'])
	# create new sequence of diagnoses
df4['SEQ_NEW'] = df4['SEQ_NEW'].fillna(df4['DX_SEQ'])
# check new item
x = df4[df4.an.isin(df_no_pdx_update.an)]
df4['DX_SEQ'] = df4['SEQ_NEW'].copy()
df4 = df4.drop(columns={'SEQ_NEW'})
df4 = df4.sort_values(['an', 'DX_SEQ'], ascending=[True, True])
df4['count'] = df4.groupby(['an'])['DX_SEQ'].transform(lambda x: sum(x == 1))
x = df4[df4['count'] == 0]
x.an.nunique()
# 24 admissions doesn't has PDx
    # drop no Pdx cases
df4 = df4[~df4.an.isin(x.an)]

    # load verified double pdx
df_double_pdx_update = pd.read_excel('review_more_than_one_pdx.xlsx', dtype={'an': str})
df_double_pdx_update = df_double_pdx_update[['an', 'DX_SEQ', 'dx', 'flag_Pdx']]

df4 = df4.merge(df_double_pdx_update, how='left', on=['an', 'DX_SEQ', 'dx'])
	# re sequence
df4.loc[(df4.flag_Pdx == 0), 'DX_SEQ'] = 2

df4 = df4.sort_values(['hn', 'date', 'an', 'DX_SEQ'],
                      ascending=[True, True, True, True])

df4.an.nunique()

    # create new seq
df4['DX_SEQ'] = df4.groupby(['an']).cumcount() + 1
df4 = df4.drop(columns={'count', 'flag_Pdx'})

del df_double_pdx_update, df_no_pdx_update
# =============================================================================
# create pdx column
# =============================================================================
    # create only pdx dataset
pdx = df4[df4.DX_SEQ == 1][['an','dx']]
pdx.columns = ['an','pdx']

pdx.an.nunique()
df4.an.nunique()

# merge pdx
df5 = df4.merge(pdx, how = 'left', on = 'an')

del df4, pdx
# =============================================================================
## collapse data
# =============================================================================
# create function collapse
header_list = pd.DataFrame([['pdx', 'first'], ['dx', lambda x: ','.join(x.astype(str))]], columns = [0,1])
header_list = dict(zip(header_list[0], header_list[1]))


df = df5.groupby(['hn', 'an', 'date', 'gender', 'age', 'biref', 'course'],
                 dropna=False).agg(header_list).reset_index()

del df5, header_list

# make a list of diagnosis codes
df['dx'] = df.dx.apply(lambda x: list(set(map(str.strip, x.split(',')))))
# remove empty list occured from row merging and column concat ([''] and '')
df['dx'] = df.dx.apply(lambda x: [var for var in x if len(var) > 0])


### find sample per year
df['year'] = pd.to_datetime(df['date'], errors = 'coerce').dt.year
    # n of admission by year
print(df['year'].value_counts().sort_index())

'''
2015    3939
2016    4360
2017    4411
2018    4200
2019    4196
'''
# =============================================================================
# create top frequency ICD-10 lists
# =============================================================================
list_icd = df[df.columns[9:]].sum(axis = 0).reset_index().sort_values(0, ascending = False)
list_icd.columns = ['code','freq']

list_top_50 = set(list(list_icd[['code']].code.head(50)))


# # save to excel
# with pd.ExcelWriter('data/top_freq_ICD/top_freq_ICD.xlsx') as writer:
#     #write each DataFrame to a specific sheet
#     pd.DataFrame(list_top_50, columns = ['icd10']).to_excel(writer, sheet_name='top_50', index = False)
 
# save complete dataset
# df.to_csv('data/top_freq_ICD/raw_data.csv', sep = '|', index = False)