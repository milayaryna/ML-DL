#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.engine import URL 


# In[2]:


SQL_SERVER = '10.11.48.11'
DATABASE = 'DBM_ExternalDB'
USERNAME = 'Potential_branch_User'
PASSWORD = 'potential_branch_user'

connection_string = "DRIVER={SQL Server}"+f";SERVER={SQL_SERVER};UID={USERNAME};PWD={PASSWORD};DATABASE={DATABASE};"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})
engine = create_engine(connection_url)


# In[3]:


Check_list = ['AY_大學基本資料', 'AY_高中基本資料', 'AY_國小基本資料', 'AY_國中基本資料', 
              'F_醫療機構基本資料','M_ATM資料', 'M_公車站點資料', 'M_火車站點資料', 'M_金融機構基本資料', 
              'M_商家基本資料', 'M_捷運站點資料', 'M_郵局據點資料', 'M_飲食店家', 'M_腳踏車站點資料','M_村里人口']

Check_list2 = ['M_六都路網', 'M_六都路網狀態及級數', 'M_二期便利商店']


# In[4]:


Current_month = str(datetime.now().month - 1)
data = {'Table': Check_list+['M_實價登錄']+Check_list2}
Check_Table = pd.read_excel("ADW_內外部Table更新狀況.xlsx")
Month = "20240" + Current_month
Month_index = Current_month+"月"


# In[5]:


for i in Check_list:
    print("Table:",i)
    sql = f"SELECT max(distinct(統計時間)) FROM [DBM_ExternalDB].[dbo].[{i}]"
    Check = pd.read_sql(sql, engine)
    if str(Check.values[0][0]) != Month:
        Check_Table[Month_index].loc[Check_Table["Table名稱"] == i] = Check.values[0]
    else:
        Check_Table[Month_index].loc[Check_Table["Table名稱"] == i] = "v"
    print("------------------\n")


# In[6]:


for i in Check_list2:
    print("Table:",i)
    sql = f"SELECT max(distinct(YYYYMM)) FROM [DBM_ExternalDB].[dbo].[{i}]"
    Check2 = pd.read_sql(sql, engine)
    if str(Check2.values[0][0]) != Month:
        Check_Table[Month_index].loc[Check_Table["Table名稱"] == i] = Check.values[0]
    else:
        Check_Table[Month_index].loc[Check_Table["Table名稱"] == i] = "v"
    print("------------------\n")


# In[7]:


sql = f"SELECT * FROM [DBM_ExternalDB].[dbo].[M_實價登錄]"
df = pd.read_sql(sql, engine)
Check3 = df.sort_values(['匯入時間'],ascending=False).groupby('匯入時間').head(1)['匯入時間'].head(1)

if Check3.values[0] != str(int(Month + "01") + 100):
    Check_Table[Month_index].loc[Check_Table["Table名稱"] == "M_實價登錄"] = Check.values[0]
else:
    Check_Table[Month_index].loc[Check_Table["Table名稱"] == "M_實價登錄"] = "v"


# In[8]:


Check_Table.to_excel("ADW_內外部Table更新狀況.xlsx",index= None)


# In[9]:


Check_Table[["Table名稱",Month_index]][0:27]


# In[ ]:




