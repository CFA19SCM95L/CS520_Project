# %%
from IPython import get_ipython

# %%
'''
Issue: Outlier, Missing value, Human error
Fix: Outlier, Missing value, Human error
Dataset is clean enough
'''


# %%
#Data preprocessing


# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium
df = pd.read_csv('processed.csv', dtype='unicode')
df.head()


# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium

df = pd.read_csv('processed.csv', dtype='unicode')
list(df.columns)
df.info()


# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium

df = pd.read_csv('processed.csv', dtype='unicode')
# replace Unknown with n.a
df.replace('UNKNOWN',np.nan, inplace=True)
df.isnull().sum().any()
df.info()


# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium


df = pd.read_csv('processed.csv', dtype='unicode')
# replace Unknown with n.a
df.replace('UNKNOWN',np.nan, inplace=True)
# Drop unnecessary columns.
df.drop(['Report Number','Local Case Number','Location', 'Vehicle Second Impact Location', 'Agency Name', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Circumstance', 'Driver Distracted By', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems', 'Off-Road Description'],axis=1 ,inplace=True)

# Columns after dropping.
df.columns


# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium


df = pd.read_csv('processed.csv', dtype='unicode')
# replace Unknown with n.a
df.replace('UNKNOWN',np.nan, inplace=True)
# Drop unnecessary columns.
df.drop(['Report Number','Local Case Number','Location', 'Vehicle Second Impact Location', 'Agency Name', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Circumstance', 'Driver Distracted By', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems', 'Off-Road Description'],axis=1 ,inplace=True)

# Change columns names. Replace spaces by underscores and upper case letters by lower case letters.
df.rename(columns=lambda x:x.replace(' ','_').lower(), inplace=True)
df.rename(columns=lambda x:x.replace('crash_date/time','date').lower(), inplace=True)
df.head()


# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium

import datetime
from dateutil.parser import parse
from dateutil import parser

df = pd.read_csv('processed.csv', dtype='unicode')
# replace Unknown with n.a
df.replace('UNKNOWN',np.nan, inplace=True)
# Drop unnecessary columns.
df.drop(['Report Number','Local Case Number','Location', 'Vehicle Second Impact Location', 'Agency Name', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Circumstance', 'Driver Distracted By', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems', 'Off-Road Description'],axis=1 ,inplace=True)

# Change columns names. Replace spaces by underscores and upper case letters by lower case letters.
df.rename(columns=lambda x:x.replace(' ','_').lower(), inplace=True)
df.rename(columns=lambda x:x.replace('crash_date/time','date').lower(), inplace=True)

df['date'] =  df['date'].apply(lambda date:parser.parse(date).strftime("%Y-%m-%d %H:%M"))
df.head()


# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium
import datetime
from dateutil.parser import parse
from dateutil import parser

df = pd.read_csv('processed.csv', dtype='unicode')
# replace Unknown with n.a
df.replace('UNKNOWN',np.nan, inplace=True)
# Drop unnecessary columns.
df.drop(['Report Number','Local Case Number','Location', 'Vehicle Second Impact Location', 'Agency Name', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Circumstance', 'Driver Distracted By', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems', 'Off-Road Description'],axis=1 ,inplace=True)

# Change columns names. Replace spaces by underscores and upper case letters by lower case letters.
df.rename(columns=lambda x:x.replace(' ','_').lower(), inplace=True)
df.rename(columns=lambda x:x.replace('crash_date/time','date').lower(), inplace=True)

df['date'] =  df['date'].apply(lambda date:parser.parse(date).strftime("%Y-%m-%d %H:%M"))

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.date.dtypes


# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium
import datetime
from dateutil.parser import parse
from dateutil import parser

df = pd.read_csv('processed.csv', dtype='unicode')
# replace Unknown with n.a
df.replace('UNKNOWN',np.nan, inplace=True)
# Drop unnecessary columns.
df.drop(['Report Number','Local Case Number','Location', 'Vehicle Second Impact Location', 'Agency Name', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Circumstance', 'Driver Distracted By', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems', 'Off-Road Description'],axis=1 ,inplace=True)

# Change columns names. Replace spaces by underscores and upper case letters by lower case letters.
df.rename(columns=lambda x:x.replace(' ','_').lower(), inplace=True)
df.rename(columns=lambda x:x.replace('crash_date/time','date').lower(), inplace=True)

df['date'] =  df['date'].apply(lambda date:parser.parse(date).strftime("%Y-%m-%d %H:%M"))

df['date'] = pd.to_datetime(df['date'], errors='coerce')

import calendar
# Number of accident per month
accidents_month = df.groupby(df['date'].dt.month).count().date

# Replace the month integers by month names.
accidents_month.index=[calendar.month_name[x] for x in range(1,13)]

accidents_month


# %%
#Data analysis


# %%
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium

df = pd.read_csv('processed.csv', dtype='unicode')
# replace Unknown with n.a
df.replace('UNKNOWN',np.nan, inplace=True)
# Drop unnecessary columns.
df.drop(['Report Number','Local Case Number','Location', 'Vehicle Second Impact Location', 'Agency Name', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Circumstance', 'Driver Distracted By', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems', 'Off-Road Description'],axis=1 ,inplace=True)
df.rename(columns=lambda x:x.replace(' ','_').lower(), inplace=True)
df.rename(columns=lambda x:x.replace('crash_date/time','date').lower(), inplace=True)
import datetime
from dateutil.parser import parse
from dateutil import parser

df['date'] =  df['date'].apply(lambda date:parser.parse(date).strftime("%Y-%m-%d %H:%M"))
df['date'] = pd.to_datetime(df['date'], errors='coerce')
# Number of accident per month
accidents_month = df.groupby(df['date'].dt.month).count().date

# Replace the month integers by month names.
accidents_month.index=[calendar.month_name[x] for x in range(1,13)]
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

# plot accidents per month
accidents_month.plot(kind='bar',figsize=(12,7), color='blue', alpha=0.5)

# title and x,y labels
plt.title('Accidents in Maryland', fontsize=20)
plt.xlabel('Month',fontsize=16)
plt.ylabel('Number of accidents',fontsize=16);

# the number of accidents decreases in Febuary, November and December. One reason could be that fewer people are driving to work in these months.


# %%
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium

df = pd.read_csv('processed.csv', dtype='unicode')
# replace Unknown with n.a
df.replace('UNKNOWN',np.nan, inplace=True)
# Drop unnecessary columns.
df.drop(['Report Number','Local Case Number','Location', 'Vehicle Second Impact Location', 'Agency Name', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Circumstance', 'Driver Distracted By', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems', 'Off-Road Description'],axis=1 ,inplace=True)
df.rename(columns=lambda x:x.replace(' ','_').lower(), inplace=True)
df.rename(columns=lambda x:x.replace('crash_date/time','date').lower(), inplace=True)
import datetime
from dateutil.parser import parse
from dateutil import parser

df['date'] =  df['date'].apply(lambda date:parser.parse(date).strftime("%Y-%m-%d %H:%M"))
df['date'] = pd.to_datetime(df['date'], errors='coerce')
# Number of accident per month
accidents_month = df.groupby(df['date'].dt.month).count().date

# Replace the month integers by month names.
accidents_month.index=[calendar.month_name[x] for x in range(1,13)]
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
# Number of accident per day of the week
accidents_day = df.groupby(df['date'].dt.dayofweek).count().date

# Replace the day integers by day names.
accidents_day.index=[calendar.day_name[x] for x in range(0,7)]

# plot accidents per day
accidents_day.plot(kind='bar',figsize=(12,7), color='magenta', alpha=0.5)

# title and x,y labels
plt.title('Accidents in Maryland', fontsize=20)
plt.xlabel('Day of the week',fontsize=16)
plt.ylabel('Number of accidents',fontsize=16);

# The number of car accidents decrease at the weekend. Weekdays present an average of 15000 car accidents per day, around 4000 more accidents than on weekends (on average 11000 car accidents per day).


# %%
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium

df = pd.read_csv('processed.csv', dtype='unicode')
# replace Unknown with n.a
df.replace('UNKNOWN',np.nan, inplace=True)
# Drop unnecessary columns.
df.drop(['Report Number','Local Case Number','Location', 'Vehicle Second Impact Location', 'Agency Name', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Circumstance', 'Driver Distracted By', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems', 'Off-Road Description'],axis=1 ,inplace=True)
df.rename(columns=lambda x:x.replace(' ','_').lower(), inplace=True)
df.rename(columns=lambda x:x.replace('crash_date/time','date').lower(), inplace=True)
import datetime
from dateutil.parser import parse
from dateutil import parser

df['date'] =  df['date'].apply(lambda date:parser.parse(date).strftime("%Y-%m-%d %H:%M"))
df['date'] = pd.to_datetime(df['date'], errors='coerce')
# Number of accident per month
accidents_month = df.groupby(df['date'].dt.month).count().date

# Replace the month integers by month names.
accidents_month.index=[calendar.month_name[x] for x in range(1,13)]
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
accidents = df.groupby(df['date'].dt.date).count().date

accidents.plot(figsize=(13,8), color='blue')

# sunday accidents
sundays = df.groupby(df[df['date'].dt.dayofweek==6].date.dt.date).count().date
plt.scatter(sundays.index, sundays, color='green', label='sunday')

# friday accidents
friday = df.groupby(df[df['date'].dt.dayofweek==4].date.dt.date).count().date
plt.scatter(friday.index, friday, color='red', label='friday')

# Title, x label and y label
plt.title('Accidents in Maryland', fontsize=20)
plt.xlabel('Date',fontsize=16)
plt.ylabel('Number of accidents per day',fontsize=16);
plt.legend()

#there are between 10–120 accidents per day and the number of accidents on friday are as a rule much higher than the number of accidents on sunday.
#The reason is people usually go to church on Sunday. On Friday, people like to hang out with their friend after work.


# %%
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium

df = pd.read_csv('processed.csv', dtype='unicode')
# replace Unknown with n.a
df.replace('UNKNOWN',np.nan, inplace=True)
# Drop unnecessary columns.
df.drop(['Report Number','Local Case Number','Location', 'Vehicle Second Impact Location', 'Agency Name', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Circumstance', 'Driver Distracted By', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems', 'Off-Road Description'],axis=1 ,inplace=True)
df.rename(columns=lambda x:x.replace(' ','_').lower(), inplace=True)
df.rename(columns=lambda x:x.replace('crash_date/time','date').lower(), inplace=True)
import datetime
from dateutil.parser import parse
from dateutil import parser

df['date'] =  df['date'].apply(lambda date:parser.parse(date).strftime("%Y-%m-%d %H:%M"))
df['date'] = pd.to_datetime(df['date'], errors='coerce')
# Number of accident per month
accidents_month = df.groupby(df['date'].dt.month).count().date

# Replace the month integers by month names.
accidents_month.index=[calendar.month_name[x] for x in range(1,13)]
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
# Number of accident per hour
accidents_hour = df.groupby(df['date'].dt.hour).count().date

# plot accidents per hour
accidents_hour.plot(kind='bar',figsize=(12,7), color='orange', alpha=0.5)

# title and x,y labels
plt.title('Accidents in Maryland', fontsize=20)
plt.xlabel('Hour',fontsize=16)
plt.ylabel('Number of accidents',fontsize=16);

#A large number of accidents occur in morning hours 7–9 and 12 - 19.


# %%
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium

df = pd.read_csv('processed.csv', dtype='unicode')
# replace Unknown with n.a
df.replace('UNKNOWN',np.nan, inplace=True)
# Drop unnecessary columns.
df.drop(['Report Number','Local Case Number','Location', 'Vehicle Second Impact Location', 'Agency Name', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Circumstance', 'Driver Distracted By', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems', 'Off-Road Description'],axis=1 ,inplace=True)
df.rename(columns=lambda x:x.replace(' ','_').lower(), inplace=True)
df.rename(columns=lambda x:x.replace('crash_date/time','date').lower(), inplace=True)
import datetime
from dateutil.parser import parse
from dateutil import parser

df['date'] =  df['date'].apply(lambda date:parser.parse(date).strftime("%Y-%m-%d %H:%M"))
df['date'] = pd.to_datetime(df['date'], errors='coerce')
# Number of accident per month
accidents_month = df.groupby(df['date'].dt.month).count().date

# Replace the month integers by month names.
accidents_month.index=[calendar.month_name[x] for x in range(1,13)]
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
# Number of accident per hour and day
accidents_hour_day = df.groupby([df['date'].dt.hour.rename('hour'),df['date'].dt.dayofweek.rename('day')]).count().date

accidents_hour_day.unstack().plot(kind='barh', figsize=(16,26))

# title and x,y labels
plt.legend(labels=[calendar.day_name[x] for x in range(0,7)],fontsize=16)
plt.title('Accidents in Maryland',fontsize=20)
plt.xlabel('Number of accidents',fontsize=16)
plt.ylabel('Hour',fontsize=16);

# there are more accidents at night on weekends than during weekdays. On the contrary, there are much more accidents from early-morning (8) til afternoon (20) during weekdays than at the weekend.


# %%
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium

df = pd.read_csv('car_crash.csv', dtype='unicode')
df.replace('UNKNOWN',np.nan, inplace=True)
df.drop(['Report Number','Local Case Number','Location', 'Vehicle Second Impact Location', 'Agency Name', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Driver At Fault', 'Circumstance', 'Driver Distracted By', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems', 'Off-Road Description'],axis=1 ,inplace=True)
df.rename(columns=lambda x:x.replace(' ','_').lower(), inplace=True)
df.rename(columns=lambda x:x.replace('crash_date/time','date').lower(), inplace=True)
import datetime
from dateutil.parser import parse
from dateutil import parser

df['date'] =  df['date'].apply(lambda date:parser.parse(date).strftime("%Y-%m-%d %H:%M"))
df['date'] = pd.to_datetime(df['date'], errors='coerce')
# Number of accident per month
accidents_month = df.groupby(df['date'].dt.month).count().date

# Replace the month integers by month names.
accidents_month.index=[calendar.month_name[x] for x in range(1,13)]
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
injuries = df.groupby(['injury_severity']).injury_severity.count().sort_values(ascending=False)

injuries.plot(kind='pie',figsize=(12,7), colors=['green','red', 'blue', 'orange', 'magenta'], labels=None, autopct='%1.1f%%', fontsize=10)

# Legend and title
plt.legend(labels=['NO APPARENT INJURY', 'POSSIBLE INJURY', 'SUSPECTED MINOR INJURY', 'SUSPECTED SERIOUS INJURY', 'FATAL INJURY'])
plt.title('Injuries in Maryland', fontsize=16)
plt.ylabel('')


# %%
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium

df = pd.read_csv('processed.csv', dtype='unicode')
# replace Unknown with n.a
df.replace('UNKNOWN',np.nan, inplace=True)
# Drop unnecessary columns.
df.drop(['Report Number','Local Case Number','Location', 'Vehicle Second Impact Location', 'Agency Name', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Circumstance', 'Driver Distracted By', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems', 'Off-Road Description'],axis=1 ,inplace=True)
df.rename(columns=lambda x:x.replace(' ','_').lower(), inplace=True)
df.rename(columns=lambda x:x.replace('crash_date/time','date').lower(), inplace=True)
import datetime
from dateutil.parser import parse
from dateutil import parser

df['date'] =  df['date'].apply(lambda date:parser.parse(date).strftime("%Y-%m-%d %H:%M"))
df['date'] = pd.to_datetime(df['date'], errors='coerce')
# Number of accident per month
accidents_month = df.groupby(df['date'].dt.month).count().date

# Replace the month integers by month names.
accidents_month.index=[calendar.month_name[x] for x in range(1,13)]
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

weather = df.groupby(['weather']).weather.count().sort_values(ascending=False)
dateLabelsFig = weather.plot(kind='bar', grid=True, figsize=(12,7), Color ='Green')

# title and x,y labels
plt.title('Accidents in Maryland', fontsize=20)
plt.xlabel('Weather',fontsize=16)
plt.ylabel('Number of accidents',fontsize=16);

# Most of accidents occur when the weather is clear. Only a few of accudents occur occur in raining day


# %%
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium

df = pd.read_csv('processed.csv', dtype='unicode')
# replace Unknown with n.a
df.replace('UNKNOWN',np.nan, inplace=True)
# Drop unnecessary columns.
df.drop(['Report Number','Local Case Number','Location', 'Vehicle Second Impact Location', 'Agency Name', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Circumstance', 'Driver Distracted By', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems', 'Off-Road Description'],axis=1 ,inplace=True)
df.rename(columns=lambda x:x.replace(' ','_').lower(), inplace=True)
df.rename(columns=lambda x:x.replace('crash_date/time','date').lower(), inplace=True)
import datetime
from dateutil.parser import parse
from dateutil import parser

df['date'] =  df['date'].apply(lambda date:parser.parse(date).strftime("%Y-%m-%d %H:%M"))
df['date'] = pd.to_datetime(df['date'], errors='coerce')
# Number of accident per month
accidents_month = df.groupby(df['date'].dt.month).count().date

# Replace the month integers by month names.
accidents_month.index=[calendar.month_name[x] for x in range(1,13)]
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

light = df.groupby(['light']).light.count().sort_values(ascending=False)
dateLabelsFig = light.plot(kind='bar', grid=True, figsize=(12,7), Color ='Green')

# title and x,y labels
plt.title('Accidents in Maryland', fontsize=20)
plt.xlabel('Light',fontsize=16)
plt.ylabel('Number of accidents',fontsize=16);

# Most car acidents occur in the morning and afternoon


# %%
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium

df = pd.read_csv('processed.csv', dtype='unicode')
# replace Unknown with n.a
df.replace('UNKNOWN',np.nan, inplace=True)
# Drop unnecessary columns.
df.drop(['Report Number','Local Case Number','Location', 'Vehicle Second Impact Location', 'Agency Name', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Circumstance', 'Driver Distracted By', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems', 'Off-Road Description'],axis=1 ,inplace=True)
df.rename(columns=lambda x:x.replace(' ','_').lower(), inplace=True)
df.rename(columns=lambda x:x.replace('crash_date/time','date').lower(), inplace=True)
import datetime
from dateutil.parser import parse
from dateutil import parser

df['date'] =  df['date'].apply(lambda date:parser.parse(date).strftime("%Y-%m-%d %H:%M"))
df['date'] = pd.to_datetime(df['date'], errors='coerce')
# Number of accident per month
accidents_month = df.groupby(df['date'].dt.month).count().date

# Replace the month integers by month names.
accidents_month.index=[calendar.month_name[x] for x in range(1,13)]
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

brand = df.groupby(['vehicle_make']).vehicle_make.count().sort_values(ascending=False)
brand = brand[:10]

dateLabelsFig = brand.plot(kind='bar', grid=True, figsize=(12,7), Color ='Green')
# title and x,y labels
plt.title('Accidents in Maryland', fontsize=20)
plt.xlabel('Brand',fontsize=16)
plt.ylabel('Number of accidents',fontsize=16);

# Most of the car accidents happen when the car is Toyota. One of the reasion is Toyota is pretty popular in Maryland.


# %%
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium

df = pd.read_csv('processed.csv', dtype='unicode')
# replace Unknown with n.a
df.replace('UNKNOWN',np.nan, inplace=True)
# Drop unnecessary columns.
df.drop(['Report Number','Local Case Number','Location', 'Vehicle Second Impact Location', 'Agency Name', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Circumstance', 'Driver Distracted By', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems', 'Off-Road Description'],axis=1 ,inplace=True)
df.rename(columns=lambda x:x.replace(' ','_').lower(), inplace=True)
df.rename(columns=lambda x:x.replace('crash_date/time','date').lower(), inplace=True)
import datetime
from dateutil.parser import parse
from dateutil import parser

df['date'] =  df['date'].apply(lambda date:parser.parse(date).strftime("%Y-%m-%d %H:%M"))
df['date'] = pd.to_datetime(df['date'], errors='coerce')
# Number of accident per month
accidents_month = df.groupby(df['date'].dt.month).count().date

# Replace the month integers by month names.
accidents_month.index=[calendar.month_name[x] for x in range(1,13)]
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

filt = df.groupby('driver_at_fault').get_group('Yes')
brand = filt.groupby(['vehicle_make']).vehicle_make.count().sort_values(ascending=False)
brand = brand[:10]
dateLabelsFig = brand.plot(kind='bar', grid=True, figsize=(12,7), Color ='Green')
# title and x,y labels
plt.title('Accidents in Maryland', fontsize=20)
plt.xlabel('Brand',fontsize=16)
plt.ylabel('Number of accidents',fontsize=16);

# Two of them are changed. Acura-> Jeep, Mercedes -> Acura. 
# People who have a Jeep car is more careless on driving than people who have a Mercedes car


# %%
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium

df = pd.read_csv('processed.csv', dtype='unicode')
# replace Unknown with n.a
df.replace('UNKNOWN',np.nan, inplace=True)
# Drop unnecessary columns.
df.drop(['Report Number','Local Case Number','Location', 'Vehicle Second Impact Location', 'Agency Name', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Circumstance', 'Driver Distracted By', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems', 'Off-Road Description'],axis=1 ,inplace=True)
df.rename(columns=lambda x:x.replace(' ','_').lower(), inplace=True)
df.rename(columns=lambda x:x.replace('crash_date/time','date').lower(), inplace=True)
import datetime
from dateutil.parser import parse
from dateutil import parser

df['date'] =  df['date'].apply(lambda date:parser.parse(date).strftime("%Y-%m-%d %H:%M"))
df['date'] = pd.to_datetime(df['date'], errors='coerce')
# Number of accident per month
accidents_month = df.groupby(df['date'].dt.month).count().date

# Replace the month integers by month names.
accidents_month.index=[calendar.month_name[x] for x in range(1,13)]
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


toyota=df.groupby('vehicle_make').get_group('TOYOTA')
accidents = toyota.groupby('vehicle_year').count().vehicle_make
accidents.plot(figsize=(13,8),color = 'blue' ,label='TOYOTA')

# Title, x label and y label
plt.title('Accidents in Maryland', fontsize=20)
plt.xlabel('Vehicle Year',fontsize=16)
plt.ylabel('Number of accidents',fontsize=16);
plt.legend()

# A dip around 2008. It could possibly be related to the 
# Great Recession that started around 2007 leading to fewer people buying new cars around then.


# %%
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium

df = pd.read_csv('processed.csv', dtype='unicode')
# replace Unknown with n.a
df.replace('UNKNOWN',np.nan, inplace=True)
# Drop unnecessary columns.
df.drop(['Report Number','Local Case Number','Location', 'Vehicle Second Impact Location', 'Agency Name', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Circumstance', 'Driver Distracted By', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems', 'Off-Road Description'],axis=1 ,inplace=True)
df.rename(columns=lambda x:x.replace(' ','_').lower(), inplace=True)
df.rename(columns=lambda x:x.replace('crash_date/time','date').lower(), inplace=True)
import datetime
from dateutil.parser import parse
from dateutil import parser

df['date'] =  df['date'].apply(lambda date:parser.parse(date).strftime("%Y-%m-%d %H:%M"))
df['date'] = pd.to_datetime(df['date'], errors='coerce')
# Number of accident per month
accidents_month = df.groupby(df['date'].dt.month).count().date

# Replace the month integers by month names.
accidents_month.index=[calendar.month_name[x] for x in range(1,13)]
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

speed = df.groupby(['speed_limit']).speed_limit.count()

dateLabelsFig = speed.plot(kind='bar', grid=True, figsize=(12,7))
plt.title('Accidents in Maryland', fontsize=20)
plt.xlabel('Speed Limit',fontsize=16)
plt.ylabel('Number of accidents',fontsize=16);


# Accidents occurs in speed between 25 - 40. People drive safely but the accident still can happen


# %%
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium

df = pd.read_csv('processed.csv', dtype='unicode')
# replace Unknown with n.a
df.replace('UNKNOWN',np.nan, inplace=True)
# Drop unnecessary columns.
df.drop(['Report Number','Local Case Number','Location', 'Vehicle Second Impact Location', 'Agency Name', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Circumstance', 'Driver Distracted By', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems', 'Off-Road Description'],axis=1 ,inplace=True)
df.rename(columns=lambda x:x.replace(' ','_').lower(), inplace=True)
df.rename(columns=lambda x:x.replace('crash_date/time','date').lower(), inplace=True)
import datetime
from dateutil.parser import parse
from dateutil import parser

df['date'] =  df['date'].apply(lambda date:parser.parse(date).strftime("%Y-%m-%d %H:%M"))
df['date'] = pd.to_datetime(df['date'], errors='coerce')
# Number of accident per month
accidents_month = df.groupby(df['date'].dt.month).count().date

# Replace the month integers by month names.
accidents_month.index=[calendar.month_name[x] for x in range(1,13)]
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

from folium import plugins

maryland_map = folium.Map(location=[39.029, -77.077], zoom_start=10)
# A mark cluster object for the car accidents
accidents = plugins.MarkerCluster().add_to(maryland_map)

# Display only accidents where serious injuries where recorded
for lat, lng, label in zip(df.latitude, df.longitude, df.injury_severity.astype(str)):
    if label!='NO APPARENT INJURY' and 'SUSPECTED MINOR INJURY':
        folium.Marker(
            location=[lat, lng],
            icon=None,
            popup=label,
        ).add_to(accidents)

maryland_map

# Most of the serious accident in Maryland occur in four area: Silver Spring, Bethesda, Gaithersburg ,and Germantown


# %%
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import folium

df = pd.read_csv('processed.csv', dtype='unicode')
# replace Unknown with n.a
df.replace('UNKNOWN',np.nan, inplace=True)
# Drop unnecessary columns.
df.drop(['Report Number','Local Case Number','Location', 'Vehicle Second Impact Location', 'Agency Name', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Circumstance', 'Driver Distracted By', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems', 'Off-Road Description'],axis=1 ,inplace=True)
df.rename(columns=lambda x:x.replace(' ','_').lower(), inplace=True)
df.rename(columns=lambda x:x.replace('crash_date/time','date').lower(), inplace=True)
import datetime
from dateutil.parser import parse
from dateutil import parser

df['date'] =  df['date'].apply(lambda date:parser.parse(date).strftime("%Y-%m-%d %H:%M"))
df['date'] = pd.to_datetime(df['date'], errors='coerce')
# Number of accident per month
accidents_month = df.groupby(df['date'].dt.month).count().date

# Replace the month integers by month names.
accidents_month.index=[calendar.month_name[x] for x in range(1,13)]
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

from folium.plugins import HeatMapWithTime

maryland_map = folium.Map(location=[39.029, -77.077], zoom_start=10)
# Nested list that contains the latitud and the longitud of the different accidents. 
hour_list = [[] for _ in range(24)]
for lat,log,hour in zip(df.latitude,df.longitude,df.date.dt.hour):
    hour_list[hour].append([lat,log]) 

# Labels indicating the hours
index = [str(i)+' Hours' for i in range(24)]

# hHat map wiht time object for the car accidents
HeatMapWithTime(hour_list, index).add_to(maryland_map)

maryland_map

# the number of accidents increases from 6 hours, remaining high until 21 hours when starts to decrease.


# %%
'''
# Conclusion
This is a great dataset after we finish the data curation. 
We can make a lot of data analysis on this dataset since there are more than 40 attributes.
We will keep working on this dataset to find more interesting analysis in the future.
'''

