#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import pymssql
import pypyodbc
import numpy as np
from os.path import join
from modules.preprocess import data_preprocess_1, data_preprocess_2_and_3
from modules.model import FECNN_train_full_data, FECNN_train_split_grid_data, FECNN_eval, FECNN_pred
from modules.post_process import create_result_table
from modules.upload_SQL import upload_grid_data, upload_pred_result
from modules.utils import ensure_folder_dir, str_time_delta

import logging
FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(filename='./save/log.txt', level=logging.INFO, filemode='a', format=FORMAT)


# # Parameters setting

# In[2]:


# Preprocess 1
YEAR_MONTH_LIST = ['202407','202408','202409','202410','202411','202412']
PRED_YEAR_MONTH = str_time_delta(YEAR_MONTH_LIST[-1], 1)
EVAL_YEAR_MONTH = YEAR_MONTH_LIST[-1]
USE_PLACE_API = True

# Preprocess 2&3
"""
MODE:
train_full_data: 
    Create training dataset(x_cnn, x_tree, y_target1, y_target2) from ALL_INSTANCE_LIST(or specified INSTANCE_LIST).
    In this mode, use training dataset to train FECNN model(auto split 20% data as validation dataset).

train_split_grid_data: 
    Create training dataset(x_cnn, x_tree, y_target1, y_target2) and \
    testing dataset(test_x_cnn, test_x_tree, test_y_target1, test_y_target2) from ALL_INSTANCE_LIST(or specified INSTANCE_LIST).
    In this mode, x_cnn, x_tree, y_target1, y_target2 will use 80% grid data to training FECNN model(auto split 20% data as validation dataset), 
    test_x_cnn, test_x_tree, test_y_target1, test_y_target2 will use 20% grid data to evaluation FECNN model. (Stratified sampling by city)
                       
eval: 'In this mode, FECNN model will not train'
    Create testing dataset(test_x_cnn, test_x_tree, test_y_target1, test_y_target2) from ALL_INSTANCE_LIST(or specified INSTANCE_LIST).
    In this mode, only use testing data to verify FECNN model.
      
pred: 'In this mode, FECNN model will not train'
    Create test_x_cnn, test_x_tree from ALL_INSTANCE_LIST(or specified INSTANCE_LIST).
    and FECNN model will predict target value (but not show any metrics such as loss).                
"""
# MODE = 'train_full_data' # options: 'train_full_data' or 'train_split_grid_data' or 'eval' or 'pred'
TARGET_2_NAME = 'Y1' # options: 'Y1' or 'Y2' or 'ATM'

MIN_MAX_SCALING = True
LOAD_SCALING_RECORD_TIME = PRED_YEAR_MONTH # If MODE is "eval" or "pred", data must standardized from scale of model training.
FEATURE_SELECTION_MODE = 'none' # options: 'none' or 'tree' or 'CNN' or 'all'
FEATURE_SELECTION_LIST = [] # If FEATURE_SELECTION_MODE not 'none', enter the features name you want to keep.

# Post-process
POST_PROCESS = True

# Upload to SQL
UPLOAD_SQL = True
SAVE_LOCAL = True


# # Data preprocess 1~3
# ## Login SQL server

# In[3]:


SQL_SERVER = '10.11.48.11'
USER_NAME = 'Potential_branch_User'
USER_PASSWORD = 'potential_branch_user'
DATABASE = 'DBM_ExternalDB'

conn = pymssql.connect(
    server=SQL_SERVER,
    user=USER_NAME,
    password=USER_PASSWORD,
    database=DATABASE
)  
cursor = conn.cursor()


# ## Data preprocess 1

# In[ ]:


class Month_city_data(object):
    def __init__(self, year_month):
        self.year_month = year_month
        self.data = {
            'Taipei': None,
            'Taoyuan': None,
            'Hsinchu': None,
            'Taichung': None,
            'Tainan': None,
            'Kaohsiung': None
        }

# Create as many Month_city_data instances as YEAR_MONTH_LIST elements number
ALL_INSTANCE_LIST = []

for year_month in YEAR_MONTH_LIST:
    ALL_INSTANCE_LIST.append(Month_city_data(year_month))

# preprocess 1 main
for month_city_instance in ALL_INSTANCE_LIST:
    year_month = month_city_instance.year_month
    folder_path = join(os.getcwd(), 'save', 'grid_data', year_month)
    ensure_folder_dir(folder_path)
    try:
        for city_name in month_city_instance.data.keys():
            csv_file_path = join(folder_path, city_name+'.csv')
            if os.path.exists(csv_file_path):
                month_city_instance.data[city_name] = pd.read_csv(csv_file_path, encoding='utf-8')
                logging.info('{}, {} grid data load from csv.'.format(year_month, city_name))
            else:
                month_city_instance.data[city_name] = data_preprocess_1(year_month, city_name, cursor, use_place_api=USE_PLACE_API)
                logging.info('Data preprocess 1 - {}, {} done.'.format(year_month, city_name))
                
                month_city_instance.data[city_name].to_csv(csv_file_path, index=False, encoding='utf-8')
    except:
        logging.exception('Catch an exception.')

cursor.close()
conn.close()


# In[ ]:


try:
    if UPLOAD_SQL:
        fcs_path = join(os.getcwd(), 'FCS', 'fcs_Data.csv')
        fcs = pd.read_csv(fcs_path, encoding = "utf-8", index_col=0)
        
        SQL_SERVER = 'DBM_Public'
        DATABASE = 'External'
        
        conn = pypyodbc.connect("DRIVER={};SERVER={};DATABASE={}".format('SQL SERVER', SQL_SERVER, DATABASE))
        cursor = conn.cursor()
        
        query = 'SELECT DISTINCT YYYYMM FROM 潛力區域預測模型_網格合併資料'
        cursor.execute(query)
        year_month_list = cursor.fetchall()
        year_month_list = [x[0] for x in year_month_list]
        
        for month_city_instance in ALL_INSTANCE_LIST:
            date = month_city_instance.year_month

            if date in year_month_list:
                logging.info('{} grid data has been uploaded to SQL server already, skip this step.'.format(date))
            else:
                for city_name in month_city_instance.data.keys():
                    df = month_city_instance.data[city_name].copy()
                    n_df = pd.merge(df, fcs, how = "left", on = 'all_grid_id')
                    upload_grid_data(cursor, '潛力區域預測模型_網格合併資料', n_df, date)
                    conn.commit()
                    logging.info('Upload {} {} grid data to SQL sever successfully.'.format(date, city_name))
                
        cursor.close()
        conn.close()
except:
    logging.exception('Catch an exception.')
    raise


# ## Data preprocess 2&3

# In[ ]:


# parameters_list = [MODE, ALL_INSTANCE_LIST, TARGET_2_NAME, MIN_MAX_SCALING, LOAD_SCALING_RECORD_TIME, 
#                    FEATURE_SELECTION_MODE, FEATURE_SELECTION_LIST]
# if MODE == 'train_full_data':
#     x_cnn, x_tree, y_target1, y_target2, _, _, _, _, _ = data_preprocess_2_and_3(parameters_list, 
#                                                                                   use_place_api=False,
#                                                                                   year_month=YEAR_MONTH)
# elif MODE == 'train_split_grid_data':
#     x_cnn, x_tree, y_target1, y_target2, \
#     test_x_cnn, test_x_tree, test_y_target1, test_y_target2, use_city_row_index = data_preprocess_2_and_3(parameters_list, 
#                                                                                   use_place_api=False,
#                                                                                   year_month=YEAR_MONTH)
# elif MODE == 'eval':
#     _, _, _, _, \
#     test_x_cnn, test_x_tree, test_y_target1, test_y_target2, use_city_row_index = data_preprocess_2_and_3(parameters_list, 
#                                                                                   use_place_api=False,
#                                                                                   year_month=YEAR_MONTH)
# else: # MODE == 'pred'
#     _, _, _, _, \
#     test_x_cnn, test_x_tree, _, _, _ = data_preprocess_2_and_3(parameters_list, 
#                                                               use_place_api=False,
#                                                               year_month=YEAR_MONTH)


# In[ ]:


# specified train INSTANCE_LIST
MODE = 'train_full_data'

parameters_list = [MODE, ALL_INSTANCE_LIST, TARGET_2_NAME, MIN_MAX_SCALING, LOAD_SCALING_RECORD_TIME, 
                   FEATURE_SELECTION_MODE, FEATURE_SELECTION_LIST]
try:
    if MODE == 'train_full_data':
        x_cnn, x_tree, y_target1, y_target2, _, _, _, _, _ = data_preprocess_2_and_3(parameters_list, 
                                                                                      use_place_api=USE_PLACE_API,
                                                                                      year_month=PRED_YEAR_MONTH)
    logging.info('Data preprocess 2&3 - Mode: train_full_data done.')
except:
    logging.exception('Catch an exception.')
    raise


# In[ ]:


print(f'shape of x_cnn: {x_cnn.shape}')
print(f'shape of x_tree: {x_tree.shape}')
print(f'shape of y_target1: {y_target1.shape}')
print(f'shape of y_target2: {y_target2.shape}')


# In[ ]:


# specified test INSTANCE_LIST (Forecast grid value for this month)
MODE = 'pred'
INSTANCE_LIST = ALL_INSTANCE_LIST[-3:]

parameters_list = [MODE, INSTANCE_LIST, TARGET_2_NAME, MIN_MAX_SCALING, LOAD_SCALING_RECORD_TIME, 
                   FEATURE_SELECTION_MODE, FEATURE_SELECTION_LIST]
try:
    if MODE == 'pred':
        _, _, _, _,         test_x_cnn, test_x_tree, _, _, _ = data_preprocess_2_and_3(parameters_list, 
                                                                    use_place_api=USE_PLACE_API, 
                                                                    year_month=PRED_YEAR_MONTH)
    logging.info('Data preprocess 2&3 - Mode: pred done.')
except:
    logging.exception('Catch an exception.')
    raise


# In[ ]:


print(f'shape of test_x_cnn: {test_x_cnn.shape}')
print(f'shape of test_x_tree: {test_x_tree.shape}')


# In[ ]:


# specified eval INSTANCE_LIST (Verify last month's model forecast performance)
MODE = 'eval'
INSTANCE_LIST = ALL_INSTANCE_LIST[-4:]

parameters_list = [MODE, INSTANCE_LIST, TARGET_2_NAME, MIN_MAX_SCALING, EVAL_YEAR_MONTH, 
                   FEATURE_SELECTION_MODE, FEATURE_SELECTION_LIST]
try:
    if MODE == 'eval':
        _, _, _, _,         eval_x_cnn, eval_x_tree, eval_y_target1, eval_y_target2,         use_city_row_index = data_preprocess_2_and_3(parameters_list, 
                                                     use_place_api=USE_PLACE_API, 
                                                     year_month=EVAL_YEAR_MONTH)
    logging.info('Data preprocess 2&3 - Mode: eval done.')
except:
    logging.exception('Catch an exception.')
    raise


# In[ ]:


print(f'shape of eval_x_cnn: {eval_x_cnn.shape}')
print(f'shape of eval_x_tree: {eval_x_tree.shape}')
print(f'shape of eval_y_target1: {eval_y_target1.shape}')
print(f'shape of eval_y_target2: {eval_y_target2.shape}')


# # Model training / eval / pred

# In[ ]:


# if MODE == 'train_full_data':
#     data = (x_cnn, x_tree, y_target1, y_target2)
#     FECNN_train_full_data(data)
# elif MODE == 'train_split_grid_data':
#     data = (x_cnn, x_tree, y_target1, y_target2, test_x_cnn, test_x_tree, test_y_target1, test_y_target2)
#     FECNN_train_split_grid_data(data)
# elif MODE == 'eval':
#     data = (test_x_cnn, test_x_tree, test_y_target1, test_y_target2)
#     pred_cls, pred_target1, pred_target2, \
#     c_attention, s_attention, cnn_filter, tree_output_weights = FECNN_eval(data, LOAD_MODEL_RECORD_TIME)
# else: # MODE == 'pred'
#     data = (test_x_cnn, test_x_tree)
#     y_pred_cls, y_pred_reg, y_pred_reg2, \
#     c_attention_list, s_attention_list, cnn_filter, tree_attention_list = FECNN_pred(data, LOAD_MODEL_RECORD_TIME)


# In[ ]:


# training model
MODE = 'train_full_data'

try:
    if MODE == 'train_full_data':
        data = (x_cnn, x_tree, y_target1, y_target2)
        FECNN_train_full_data(data, save_year_month=PRED_YEAR_MONTH)
    logging.info('FECNN model training done.')
except:
    logging.exception('Catch an exception.')
    raise


# In[ ]:


# model predition
MODE = 'pred'

try:
    if MODE == 'pred':
        data = (test_x_cnn, test_x_tree)
        pred_cls, pred_target1, pred_target2,         c_attention, s_attention, cnn_filter, tree_output_weights = FECNN_pred(data, PRED_YEAR_MONTH)
    logging.info('FECNN Model prediction complete.')
except:
    logging.exception('Catch an exception.')
    raise


# In[ ]:


# model evaluation
MODE = 'eval'

try:
    if MODE == 'eval':
        data = (eval_x_cnn, eval_x_tree, eval_y_target1, eval_y_target2)
        _, _, _, _, _, _, _,         target1_mse, target1_mae, target2_mse, target2_mae = FECNN_eval(data, EVAL_YEAR_MONTH, use_city_row_index)
    logging.info('Model evaluation for last month grid data complete.')
    logging.info('Target1: ')
    logging.info(f'MSE: {target1_mse}')
    logging.info(f'MAE: {target1_mae}')
    logging.info('Target2: ')
    logging.info(f'MSE: {target2_mse}')
    logging.info(f'MAE: {target2_mae}')
except:
    logging.exception('Catch an exception.')
    raise


# # Data post-process

# In[ ]:


try:
    if POST_PROCESS:
        pred_data = [pred_target1, pred_target2, c_attention, s_attention, tree_output_weights]
        truth_data = [eval_y_target1, eval_y_target2] # Use the ground truth from the previous month
        result = create_result_table(x_tree=test_x_tree, 
                                     pred_data=pred_data,
                                     truth_data=truth_data,
                                     load_model_record_time=PRED_YEAR_MONTH)
        logging.info('Data post-process done.')
except:
    logging.exception('Catch an exception.')
    raise


# In[ ]:


result


# # Upload prediction result to SQL server

# In[ ]:


try:
    if UPLOAD_SQL:
        SQL_SERVER = 'DBM_Public'
        DATABASE = 'External'

        conn = pypyodbc.connect("DRIVER={};SERVER={};DATABASE={}".format('SQL SERVER', SQL_SERVER, DATABASE))
        cursor = conn.cursor()

        upload_pred_result(cursor, '潛力區域預測模型_預測結果', result, PRED_YEAR_MONTH, SAVE_LOCAL)

        conn.commit()
        cursor.close()
        conn.close()
        
        logging.info('Upload SQL sever done.')
except:
    logging.exception('Catch an exception.')
    raise


# In[ ]:




