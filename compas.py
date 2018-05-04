# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 13:21:49 2018

@author: TapperR
"""

import os
import pandas as pd

#os.chdir('C:\\Users\\Robin\\Desktop\\compas-analysis')
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
#from datetime import date
#from datetime import datetime
#from datetime import timedelta
#
#date_format = "%Y-%m-%d %H:%M:%S"
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



#is there any correlation between the length of jail time and the decile score? yes, but little
df['length_of_stay'] = (df['c_jail_out'].subtract(df['c_jail_in'])).astype('timedelta64[D]') + 1
df['length_of_stay'].corr(df['decile_score'])


#how is the distribution of the age look like
df['age_cat'].value_counts()

#how is the distribution of the race look like
df['race'].value_counts()
print("Black defendants: %.2f%%" %            (3175 / 6172 * 100))  #maybe it is possible to implement a for-loop in here
print("White defendants: %.2f%%" %            (2103 / 6172 * 100))
print("Hispanic defendants: %.2f%%" %         (509  / 6172 * 100))
print("Asian defendants: %.2f%%" %            (31   / 6172 * 100))
print("Native American defendants: %.2f%%" %  (11   / 6172 * 100))

#how is the distribution of the score look like
df['score_text'].value_counts()
df['c_charge_desc'].value_counts()

#contingency table creation for the factors gender and race
pd.crosstab(index = df['sex'], columns = df['race'])


#how is the distribution of the age look like
df['sex'].value_counts()
print("Men: %.2f%%" %   (4997 / 6172 * 100))
print("Women: %.2f%%" % (1175 / 6172 * 100))


#how many offenders were recidivist (2809, 45.51%)
a = len(df[df['two_year_recid'] == 1])
a/len(df)*100


#is there a downward trend in decile scores in the enitre data
plt.bar(range(len(df['decile_score'].unique())), df['decile_score'].value_counts().sort_index(), align='center', alpha=1, color = 'black')


#is there a downward trend in decile scores in the data, which contains only black people (not that clear)
plt.bar(np.arange(1, len(df[df['race'] == 'African-American']['decile_score'].unique())+1,1), df[df['race'] == 'African-American']['decile_score'].value_counts().sort_index(), align='center', alpha=1, color = 'black')

#is there a downward trend in decile scores in the data, which contains only white people (clear)
plt.bar(np.arange(1, len(df[df['race'] == 'Caucasian']['decile_score'].unique())+1,1), df[df['race'] == 'Caucasian']['decile_score'].value_counts().sort_index(), align='center', alpha=1, color = 'blue')









########## Doing some regression, factoring, etc ##############


#creating some factors in the data for some logistic regressions
#df['crime_factor'] = df['c_charge_degree'].astype('category')
#dummies0 = pd.get_dummies(df['crime_factor'])
#df = pd.concat([df, dummies0], axis=1)
#df = df.drop(['crime_factor', 'F'], axis=1)




#df['age_factor'] = df['age_cat'].astype('category')
#
##within(age_factor <- relevel(age_factor, ref = 1)), relevel, which means making a reference value for the regression
#dummies1 = pd.get_dummies(df['age_factor'])
#df = pd.concat([df, dummies1], axis=1)
#df = df.drop(['age_factor', '25 - 45'], axis=1)





#df['race_factor'] = df['race'].astype('category')
#
##relevel, which means making a reference value for the regression (as above with ref = 3 (Caucasian))
#dummies2 = pd.get_dummies(df['race_factor'])
#df = pd.concat([df, dummies2], axis=1)
#df = df.drop(['race_factor', 'Caucasian'], axis=1)


#gender factor
#df['gender_factor'] = df['sex'].astype('category')


#dummies2 = pd.get_dummies(df['gender_factor'])
#df = pd.concat([df, dummies2], axis=1)
#df = df.drop(['gender_factor', 'Male'], axis=1)


#score_factor with only two levels
df['score_factor'] = pd.Series('nan')
df['score_factor'][df['score_text'] == 'Low']  = 'LowScore'
df['score_factor'][df['score_text'] != 'Low']  = 'HighScore'
df['score_factor'] = df['score_factor'].astype('category')

#dummies3 = pd.get_dummies(df['score_factor'])
#df = pd.concat([df, dummies2], axis=1)





####### logistic regression #########

import statsmodels.api as sm
import statsmodels.formula.api as smf

#data = smf.datasets.scotland.load()
#data.exog = smf.add_constant(data.exog)
#data.endog

#model1 = smf.GLM(df['score_factor'],  [df['Female'], df['Greater than 45'], df['Less than 25'], df['African-American'], df['Asian'], df['Hispanic'], df['Native American'], df['Other'], df['Greater than 45'], df['priors_count'], df['M'], df['two_year_recid']], family = sm.families.Binomial())
#model1 = smf.GLM([df["score_factor"]],  [df["Female"]], family = sm.families.Binomial())


#### Test the difference ####
#model2 = smf.glm(formula = "decile_score ~ Female + C(age_cat) + C(race) + priors_count + C(c_charge_degree) + two_year_recid",  data = df)

#reorder for getting another reference level
df['sex'] = df['sex'].astype('category')
df['sex'] = df['sex'].cat.reorder_categories(['Male', 'Female'])

df['race'] = df['race'].astype('category')
df['race'] = df['race'].cat.reorder_categories(['Caucasian', 'African-American', 'Asian', 'Hispanic', 'Native American', 'Other'])


model2 = smf.glm(formula = "C(score_factor) ~ C(sex) + C(age_cat) + C(race) + priors_count + C(c_charge_degree) + two_year_recid",  data = df, family = sm.families.Binomial())

res = model2.fit()
print(res.summary())


##### Interpreting the GLM-Result ######
#how the probability of a high score increase, if the offender is black
control = np.exp(-1.52554) / (1 + np.exp(-1.52554))
np.exp(0.47721) / (1 - control + (control * np.exp(0.47721)))

#how the probability of a high score increase, if the offender is a woman
control = np.exp(-1.52554) / (1 + np.exp(-1.52554))
np.exp(0.2213) / (1 - control + (control * np.exp(0.2213)))

#how the probability of a high score increase, if the offender is a woman
np.exp(0.2689) / (1 - control + (control * np.exp(0.2689)))


#df["score_factor"].dtypes




#df = sm.datasets.get_rdataset("Guerry", "HistData").data

#df = df[['Lottery', 'Literacy', 'Wealth', 'Region']].dropna()



#predictors =  [df["Female"]]
#response_col = [df["score_factor"]]
#glm_model = H2OGeneralizedLinearEstimator(family= "binomial", lambda_ = 0, compute_p_values = True)
#glm_model.train(predictors, response_col)
#
#
#
#import h2o
#h2o.init()
#from h2o.estimators.glm import H2OGeneralizedLinearEstimator
#
#prostate = h2o.import_file("https://h2o-public-test-data.s3.amazonaws.com/smalldata/prostate/prostate.csv")
#prostate['CAPSULE'] = prostate['CAPSULE'].asfactor()
#prostate['RACE'] = prostate['RACE'].asfactor()
#prostate['DCAPS'] = prostate['DCAPS'].asfactor()
#prostate['DPROS'] = prostate['DPROS'].asfactor()
#
#
#predictors = ["AGE", "RACE", "VOL", "GLEASON"]
#response_col = "CAPSULE"
#
#glm_model = H2OGeneralizedLinearEstimator(family= "binomial", lambda_ = 0, compute_p_values = True)
#glm_model.train(predictors, response_col, training_frame= prostate)
#print(glm_model.coef())






##### Risk of Violent Recidivism #########

raw_data = raw3
len(raw_data)

df2 = raw_data[raw_data['days_b_screening_arrest']<=30]    #charge date of defendate have to be 30 days before or after the defendants arrest
df2 = df2[df2['days_b_screening_arrest']>=-30]

df2 = df2[df2['is_recid']!=-1]   #is_recid is -1, if there was no compas case at all for the defendant

df2 = df2[df2['c_charge_degree']!='O']   #traffic offenses, that will not end up in jail are not regarded


#raw_data2 = raw_data[np.isnan(raw_data['c_arrest_date'])==False]
#raw_data2 = raw_data[raw_data['c_arrest_date'].isnull()==False]

df2 = df2[df2['v_score_text'].isnull()==False]  #is there any missing score
len(df2)

#how is the distribution of the score look like
df2['v_score_text'].value_counts()


#how many offenders were recidivist (652, 16.21%)
a = len(df2[df2['two_year_recid'] == 1])
a/len(df2)*100



#is there a downward trend in decile scores in the enitre data
plt.bar(np.arange(1, len(df2['v_decile_score'].unique())+1,1), df2['v_decile_score'].value_counts().sort_index(), align='center', alpha=1, color = 'black')


#is there a downward trend in decile scores in the data, which contains only black people (not that clear)
plt.bar(np.arange(1, len(df2[df2['race'] == 'African-American']['v_decile_score'].unique())+1,1), df2[df2['race'] == 'African-American']['v_decile_score'].value_counts().sort_index(), align='center', alpha=1, color = 'black')

#is there a downward trend in decile scores in the data, which contains only white people (clear)
plt.bar(np.arange(1, len(df2[df2['race'] == 'Caucasian']['v_decile_score'].unique())+1,1), df2[df2['race'] == 'Caucasian']['v_decile_score'].value_counts().sort_index(), align='center', alpha=1, color = 'blue')



#releving
df2['score_factor'] = pd.Series('nan')
df2['score_factor'][df2['v_score_text'] == 'Low']  = 'LowScore'
df2['score_factor'][df2['v_score_text'] != 'Low']  = 'HighScore'
df2['score_factor'] = df2['score_factor'].astype('category')


df2['sex'] = df2['sex'].astype('category')
df2['sex'] = df2['sex'].cat.reorder_categories(['Male', 'Female'])

df2['race'] = df2['race'].astype('category')
df2['race'] = df2['race'].cat.reorder_categories(['Caucasian', 'African-American', 'Asian', 'Hispanic', 'Native American', 'Other'])


model2 = smf.glm(formula = "C(score_factor) ~ C(sex) + C(age_cat) + C(race) + priors_count + C(c_charge_degree) + two_year_recid",  data = df, family = sm.families.Binomial())

res = model2.fit()
print(res.summary())


##### Interpreting the GLM-Result ######
#how the probability of a high score increase, if the offender is black
control = np.exp(-2.2427 ) / (1 + np.exp(-2.2427 ))
np.exp(0.6589) / (1 - control + (control * np.exp(0.6589)))

#how the probability of a high score increase, if the offender is a woman
np.exp(-0.7289) / (1 - control + (control * np.exp(-0.7289)))

#how the probability of a high score increase, if the offenders age is less than 25
np.exp(3.1459) / (1 - control + (control * np.exp(3.1459)))








###############   THE ESSENTIAL: Predictive Accuracy of COMPAS ##############

df3 = raw4
df3 = df3[df3['score_text'].isnull()==False]
df3 = df3[df3['end']>df3['start']]
len(df3)

df3['race_factor'] = df3['race'].astype('category')
df3['race_factor'] = df3['race_factor'].cat.reorder_categories(['Caucasian', 'African-American', 'Asian', 'Hispanic', 'Native American', 'Other'])


df3['score_factor'] = df3['score_text'].astype('category')
df3['score_factor'] = df3['score_factor'].cat.reorder_categories(['Low', 'High', 'Medium'])



#df3.reset_index(inplace=True, drop = True)
#df3.drop_duplicates(subset = 'id', inplace = True)
#df3.reset_index(inplace=True, drop = True)


df3['score_factor']
df3['race_factor'].value_counts()

 
#from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter
from lifelines.estimation import KaplanMeierFitter


cph = CoxPHFitter()

df3['duration'] = df3['end'] - df3['start']

#cph.fit(df3[['duration', 'event']], duration_col = df3['duration'], event_col = df3['event'])
#cph.fit(df3, duration_col = df3['duration'], event_col = df3['event'])
#
#
#cph.fit()
#
#
#kmf = KaplanMeierFitter()
#
#
#kmf.fit(df3['duration'], event_observed = df3['event'])
#
#kmf.survival_function_
#kmf.median_
#kmf.plot()
#
#
#from lifelines.datasets import load_rossi
#
#rossi_dataset = load_rossi()
#cph.fit(rossi_dataset, duration_col='week', event_col='arrest')
#
#cph.print_summary()
#
#
#
##### trying some plotly
#
#%load_ext rpy2.ipython
#%R install.packages("devtools")
#%R devtools::install_github("ropensci/plotly")
#%R install.packages("OIsurv")
#
#
#import numpy as np
#import pandas as pd
#import lifelines as ll
#
#
#from IPython.display import HTML
#%matplotlib inline
#import matplotlib.pyplot as plt
#import plotly.plotly as py
#import plotly.tools as tls   
#from plotly.graph_objs import *
#
#from pylab import rcParams
#rcParams['figure.figsize']=10, 5
#
#
#
#tongue = pd.read_table('tongue.csv', sep=',', encoding='utf-8',  na_filter = True)
#tongue.head()
#
#print(tongue['time'].mean())






<<<<<<< HEAD
######## THIS IS TO SOLVE ########
=======
######### THIS IS TO SOLVE ########
>>>>>>> dce054d18f9ae86316456874d9b818526e3e822f
#f <- Surv(start, end, event, type="counting") ~ score_factor
#model <- coxph(f, data=data)
#summary(model)
#
#df3['start'], df3['end'], df3['event'], df3['score_factor']

##################################


#from lifelines.estimation import KaplanMeierFitter
#kmf = KaplanMeierFitter()


#f = tongue.type == 1
#T = tongue[f]['time']
#C = tongue[f]['delta']
#
#kmf.fit(T, event_observed=C)
#kmf.plot(title='Tumor DNA Profile 1')



#df3['duration'] = df3['end'] - df3['start']


#kmf.fit(df3['duration'], event_observed = df3['event'])
#kmf.plot()
#
#
#
#
#cph = CoxPHFitter()
##cph.fit(df3[['duration', 'event']], duration_col = df3['duration'], event_col = df3['event'])
#cph.fit(df = df3[['duration', 'event', 'High', 'Medium']], duration_col = 'duration', event_col = 'event')
#cph.print_summary()
#




#from lifelines.datasets import load_regression_dataset
#regression_dataset = load_regression_dataset()
#
#regression_dataset.head()
#
#cph.fit(regression_dataset, 'T', event_col='E')


#creating dummies for the score factor for the survival analysis
dummies0 = pd.get_dummies(df3['score_factor'])
df3 = pd.concat([df3, dummies0], axis=1)
#df3 = df3.drop(['score_factor', 'Low'], axis=1)


##### how good is the categorization in high, medium and low
cph = CoxPHFitter()
cph.fit(df = df3[['duration', 'event', 'High', 'Medium']], duration_col = 'duration', event_col = 'event')
cph.print_summary()
cph.plot()

cph.fit(df = df3[['duration', 'event', 'High', 'Medium']], duration_col = 'duration', event_col = 'event')
cph.predict_survival_function()







#how good is the numeric decile_score
df4 = df3[['duration', 'event', 'decile_score']]

cph.fit(df = df4, duration_col = 'duration', event_col = 'event')
cph.print_summary()
#cph.plot()
cph.predict_survival_function(X = df4)




#Compare less cases to compute the concordance, maybe the first 20, like it is described in 
#http://dni-institute.in/blogs/model-performance-assessment-statistics-concordance-steps-to-calculate/


dummies0 = pd.get_dummies(df3['race_factor'])
df3 = pd.concat([df3, dummies0], axis=1)
#df3 = df3.drop(['race_factor', 'Caucasian'], axis=1)



#creating interaction terms between race and score factor first
#from sklearn.preprocessing import PolynomialFeatures
#poly = PolynomialFeatures(interaction_only=True,include_bias = False)
#
#poly.fit_transform(df3[['race_factor', 'score_factor']])
#
#dummie1 = PolynomialFeatures()

for i in df3['race_factor'].unique():
    if i != 'Caucasian':
        for j in ['High', 'Medium']:
            df3['%s_%s'%(i,j)] = df3['%s'%i]*df3['%s'%j]




df5 = df3[['duration', 'event', 'African-American', 'Asian', 'Hispanic', 'Native American', 'Other', 'High', 'Medium', 'African-American_High', 'African-American_Medium', 'Asian_High', 'Asian_Medium', 'Hispanic_High', 'Hispanic_Medium', 'Native American_High', 'Native American_Medium', 'Other_High', 'Other_Medium']]

cph.fit(df = df5, duration_col = 'duration', event_col = 'event')
cph.print_summary()
#cph.plot()








import math
print("Black High Hazard: %.2f" % (math.exp(-0.1864 + 1.1478)))
print("White High Hazard: %.2f" % (math.exp(1.1478)))
print("Black Medium Hazard: %.2f" % (math.exp(0.7736-0.1694)))
print("White Medium Hazard: %.2f" % (math.exp(0.7736)))


df5 = df3[df3['']
df5 = df3[['duration', 'event']]

cph.fit(df = df5, duration_col = 'duration', event_col = 'event')
cph.predict_survival_function(X = df5).plot()




#Kaplan Meier plots

from lifelines.estimation import KaplanMeierFitter
kmf = KaplanMeierFitter()


df6 = df3[['duration', 'event']]
kmf.fit(df6['duration'],df6['event'])
kmf.plot()


#how does the survival curve look alike for black people
df6a = df3[df3['race_factor'] == 'African-American']
df6a = df6a[df6a['score_factor'] == 'Low']
df6b = df6a[['duration', 'event']]
kmf.fit(df6b['duration'],df6b['event'])
kmf.plot()


#how does the survival curve look alike for white people
df6c = df3[df3['race_factor'] == 'Caucasian']
df6c = df6c[df6c['score_factor'] == 'Low']
df6d = df6c[['duration', 'event']]
kmf.fit(df6d['duration'],df6d['event'])
kmf.plot()








#how high is the difference between black and white regarding the concordance
cph = CoxPHFitter()
df3a = df3[df3['race_factor'] == 'African-American']
cph.fit(df = df3a[['duration', 'event', 'High', 'Medium']], duration_col = 'duration', event_col = 'event')
cph.print_summary()


df3a = df3[df3['race_factor'] == 'Caucasian']
cph.fit(df = df3a[['duration', 'event', 'High', 'Medium']], duration_col = 'duration', event_col = 'event')
cph.print_summary()



##########     Directions of the Racial Bias    #############

from truth_tables import PeekyReader, Person, table, is_race, count, vtable, hightable, vhightable
from csv import DictReader


people = []

f = open('cox-parsed.csv', 'r')

with open('cox-parsed.csv') as f:
    #print(f)
    reader = PeekyReader(DictReader(f))
    try:
        while True:
            p = Person(reader)
            if p.valid:
                people.append(p)
    except StopIteration:
        pass
    
    
#len(people)
#people[0].score_valid
#people[0].rows
#people[0].valid
#people[0].__rows


    
pop = list(filter(lambda i: ((i.recidivist == True and i.lifetime <= 730) or
                              i.lifetime > 730), list(filter(lambda x: x.score_valid, people))))



#len(list(filter(lambda x: x.score_valid, people)))
#len(pop)    


recid = list(filter(lambda i: i.recidivist == True and i.lifetime <= 730, pop))
#len(recid) 



rset = set(recid)
surv = [i for i in pop if i not in rset]



print("All defendants")
table(list(recid), list(surv))




#### Can we beat it with a Neural Network???? ########


import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

#start a tensorflow program
sess = tf.Session()
sess.run(hello)

a = tf.constant(10)
b = tf.constant(32)

sess.run(a + b)



#end a tensorflow program
sess.close()
























from truth_tables import PeekyReader, Person, table, is_race, count, vtable, hightable, vhightable
from csv import DictReader

people = []
with open("cox-parsed.csv") as f:
    reader = PeekyReader(DictReader(f))
    try:
        while True:
            p = Person(reader)
            if p.valid:
                people.append(p)
    except StopIteration:
        pass


#lifetime = start - end, start = out_custody - in_custody ||| end = r_offense_date - in_custody


pop = list(filter(lambda i: ((i.recidivist == True and i.lifetime <= 730) or
                              i.lifetime > 730), list(filter(lambda x: x.score_valid, people))))


recid = list(filter(lambda i: i.recidivist == True and i.lifetime <= 730, pop))
#non_recid = list(filter(lambda i: i.lifetime > 730, pop))

rset = set(recid)
#len(rset)


surv = [i for i in pop if i not in rset]


print("All defendants")
table(list(recid), list(surv))





import statistics
print("Average followup time %.2f (sd %.2f)" % (statistics.mean(map(lambda i: i.lifetime, pop)),
                                                statistics.stdev(map(lambda i: i.lifetime, pop))))
print("Median followup time %i" % (statistics.median(map(lambda i: i.lifetime, pop))))







print("Black defendants")
is_afam = is_race("African-American")
table(list(filter(is_afam, recid)), list(filter(is_afam, surv)))



print("White defendants")
is_white = is_race("Caucasian")
table(list(filter(is_white, recid)), list(filter(is_white, surv)))








hightable(list(filter(is_white, recid)), list(filter(is_white, surv)))
hightable(list(filter(is_afam, recid)), list(filter(is_afam, surv)))










########   violent crime   #######


from truth_tables import PeekyReader, Person, table, is_race, count, vtable, hightable, vhightable
from csv import DictReader



vpeople = []
with open("cox-violent-parsed.csv") as f:
    reader = PeekyReader(DictReader(f))
    try:
        while True:
            p = Person(reader)
            if p.valid:
                vpeople.append(p)
    except StopIteration:
        pass

vpop = list(filter(lambda i: ((i.violent_recidivist == True and i.lifetime <= 730) or
                              i.lifetime > 730), list(filter(lambda x: x.vscore_valid, vpeople))))
vrecid = list(filter(lambda i: i.violent_recidivist == True and i.lifetime <= 730, vpeople))
vrset = set(vrecid)
vsurv = [i for i in vpop if i not in vrset]


print("All defendants")
vtable(list(vrecid), list(vsurv))




print("Black defendants")
is_afam = is_race("African-American")
vtable(list(filter(is_afam, vrecid)), list(filter(is_afam, vsurv)))


















