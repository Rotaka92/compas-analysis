# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 13:21:49 2018

@author: TapperR
"""

import os
import pandas as pd

os.chdir('C:\\Users\\TapperR\\Desktop\\compas\\compas-analysis')

raw = pd.read_table('compas-scores-raw.csv', sep=',', encoding='utf-8',  na_filter = True)
raw1 = pd.read_table('compas-scores.csv', sep=',', encoding='utf-8',  na_filter = True)
raw2 = pd.read_table('compas-scores-two-years.csv', sep=',', encoding='utf-8',  na_filter = True)
raw3 = pd.read_table('compas-scores-two-years-violent.csv', sep=',', encoding='utf-8',  na_filter = True)
raw4 = pd.read_table('cox-parsed.csv', sep=',', encoding='utf-8',  na_filter = True)
raw5 = pd.read_table('cox-violent-parsed.csv', sep=',', encoding='utf-8',  na_filter = True)
raw6 = pd.read_table('cox-violent-parsed_filt.csv', sep=',', encoding='utf-8',  na_filter = True)

#how many people we have 
len(raw['Person_ID'].unique())

#look up the data
raw.head()
raw1.head() 
raw2.head() #similar data as raw1 except the additional variables describing if any recidivism happened in the two years after jail out
                #question is, that 'start' and 'end' and 'event' and 'recid' and 'recid.1' means (last 5 variables)
raw3.head() #similar data as raw1 except the additional variables describing if any violent-recidivism happened in the two years after jail out




raw_data = raw2
len(raw_data)

###########

df = raw_data[raw_data['days_b_screening_arrest']<=30]    #charge date of defendate have to be 30 days before or after the defendants arrest
df = df[df['days_b_screening_arrest']>=-30]

df = df[df['is_recid']!=-1]   #is_recid is -1, if there was no compas case at all for the defendant

df = df[df['c_charge_degree']!='O']   #traffic offenses, that will not end up in jail are not regarded


#raw_data2 = raw_data[np.isnan(raw_data['c_arrest_date'])==False]
#raw_data2 = raw_data[raw_data['c_arrest_date'].isnull()==False]

df = df[df['score_text'].isnull()==False]  #is there any missing score
len(df)

#############
from datetime import date
from datetime import datetime
from datetime import timedelta

date_format = "%Y-%m-%d %H:%M:%S"
#a = datetime.strptime(df['c_jail_out'], date_format)

#df.reset_index(inplace=True, drop = True)

#i = 0
#for j in range(len(df)):
#    # j=0
#    df['length_of_stay'][j] =  (datetime.strptime(df['c_jail_out'][j], date_format).date() - datetime.strptime(df['c_jail_in'][j], date_format).date()).days
#    i += 1
#    print(i)
#
#year = timedelta(days=365)
#year.total_seconds()

#print(_)


#a = date(df['c_jail_out'][0])
#b = date(df['c_jail_in'][0])
#
#a-b.days


df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])



#(df['c_jail_out'] - df['c_jail_in'])
#
#
#(df['c_jail_out'] - df['c_jail_in']) / np.timedelta64(-1, 'D')



#is there any correlation between the length of jail time and the decile score?
df['length_of_stay'] = (df['c_jail_out'].subtract(df['c_jail_in'])).astype('timedelta64[D]') + 1
df['length_of_stay'].corr(df['decile_score'])









#### from the original script with cell magic
%load_ext rpy2.ipython
import warnings
warnings.filterwarnings('ignore')


%%R
library(dplyr)
library(ggplot2)
setwd("C:/Users/TapperR/Desktop/compas/compas-analysis")
raw_data <- read.csv("compas-scores-two-years.csv")
%R nrow(raw_data)


%%R
df <- dplyr::select(raw_data, age, c_charge_degree, race, age_cat, score_text, sex, priors_count, 
                    days_b_screening_arrest, decile_score, is_recid, two_year_recid, c_jail_in, c_jail_out) %>% 
        filter(days_b_screening_arrest <= 30) %>%
        filter(days_b_screening_arrest >= -30) %>%
        filter(is_recid != -1) %>%
        filter(c_charge_degree != "O") %>%
        filter(score_text != 'N/A')
nrow(df)

%R nrow(df)


%%R
%R df$length_of_stay <- as.numeric(as.Date(df$c_jail_out) - as.Date(df$c_jail_in))
%R df$length_of_stay[1:10]
%R cor(df$length_of_stay, df$decile_score)
%R print(df$length_of_stay)
