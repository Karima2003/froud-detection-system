#!/usr/bin/env python
# coding: utf-8

# Importing necessary library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings 
warnings.filterwarnings('ignore')



train_path = '/home/aya/Streaming-Fraud-Detection/data/raw/fraudTrain.csv'
test_path = '/home/aya/Streaming-Fraud-Detection/data/raw/fraudTest.csv'


# In[3]:


df_train = pd.read_csv(train_path, low_memory=False, index_col=0)
df_test = pd.read_csv(test_path, low_memory=False, index_col=0)
# Merge train and test dataset
df = pd.concat([df_train, df_test], ignore_index=True)


# In[4]:


df.head()


# In[5]:


print(f"👉 Shape (Rows, Columns): {df.shape}\n")
print(f"👉 Data Types: {df.dtypes}\n")


# * cc_num: (Credit Card Number) The customer's credit card number (likely encrypted or masked).
# * merchant: Merchant Name where the transaction took place.
# * category: Category of the merchant (e.g., restaurant, supermarket, electronics, etc.).
# * amt: Transaction amount (possibly in a specific currency).
# * first:	First name of the cardholder.
# * last:	Last name of the cardholder.
# * gender:	Gender of the cardholder (Male or Female).
# * street: Street address of the cardholder.
# * city: City where the cardholder resides.
# * state	: State/Province where the cardholder resides.
# * zip: Zip code of the cardholder's address.
# * lat: Latitude of the cardholder's address.
# * long: Longitude of the cardholder's address.
# * city_pop: Population of the city where the cardholder resides.
# * job: Occupation of the cardholder.
# * dob: Date of birth of the cardholder.
# * trans_num: Transaction ID (a unique identifier for each transaction).
# * unix_time: Transaction timestamp in Unix time format (seconds since January 1, 1970).
# * merch_lat: Latitude of the merchant location.
# * merch_long: Longitude of the merchant location.

# In[6]:


# Transfer trans_time and dob to datetime type
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["dob"] = pd.to_datetime(df["dob"])
df.head()


# In[7]:


# Summary Dataset
def dataframe_summary(df):
    """Provides a concise summary of a Pandas DataFrame."""

    print("\n🔹 DataFrame Summary 🔹")
    print("=" * 50)

    print(f"📌 Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    print("📌 Data Types:\n", df.dtypes, "\n")

    missing_values = df.isnull().sum()
    print("📌 Missing Values:\n", missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values.", "\n")
    print("📌 Duplicate rows in train_data:", df.duplicated().sum())
    print("📌 Statistics:\n", df.describe().T, "\n")
    print("📌 Unique Values:\n", df.nunique(), "\n")
    print("📌 Memory Usage:\n", df.memory_usage(deep=True), "\n")
    print("📌 Sample Data:\n", df.head(), "\n")

    print("=" * 50)


# In[8]:


dataframe_summary(df)


# # Explorantory Data Analysis (EDA)

# ### 1. Data imbalance check Fraud and Non Fraud Transaction

# In[9]:


df.is_fraud.value_counts()


# In[11]:


fraud_counts = df['is_fraud'].value_counts()

labels = ['Not Fraud (0)', 'Fraud (1)']
colors = ['lightblue', 'red']

plt.figure(figsize=(6, 6))
plt.pie(fraud_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, wedgeprops={'edgecolor': 'black'})

# Tiêu đề
plt.title('Percentage of Fraudulent and Non-Fraudulent Transactions')
# Sauvegarde du graphique dans un fichier
plt.savefig('/home/aya/Streaming-Fraud-Detection/resultats/fraud_percentage_pie_chart.png')



# Don't use accuracy score as a metric with imbalanced datasets - it will be usually high and misleading. In this dataset:
# 
# we have 99.5% of Normal transactions and only 0,5% of fraud transactions;
# 
# whis means that a blind guess (bet on Normal) would give us accuracy of 99,5%.

# ### 2. Transaction amount vs Fraud

# In[12]:


df['amt'].describe()


# In[13]:


np.percentile(df['amt'],99)


# In[14]:


#amount vs fraud
import seaborn as sns
ax=sns.histplot(x='amt',data=df[df.amt<=1000],hue='is_fraud',stat='percent',multiple='dodge',common_norm=False,bins=25)
ax.set_ylabel('Percentage in Each Type')
ax.set_xlabel('Transaction Amount in USD')
plt.legend(title='Type', labels=['Fraud', 'Not Fraud'])
plt.tight_layout()
plt.savefig('/home/aya/Streaming-Fraud-Detection/resultats/amount_vs_fraud.png')


# - Mid-to-high value transactions ($200 - $1000) have a higher fraud risk.
# - Small transactions (< $100) are the most common, but fraud still occurs at these levels.
# - Implement stricter monitoring on transactions above $200 to enhance fraud detection. 🚀

# ### 3. Gender vs Fraud

# In[15]:


#Gender vs Fraud
ax=sns.histplot(x='gender',data=df, hue='is_fraud',stat='percent',multiple='dodge',common_norm=False)
ax.set_ylabel('Percentage')
ax.set_xlabel('Credit Card Holder Gender')
plt.legend(title='Type', labels=['Fraud', 'Not Fraud'])
plt.tight_layout()
plt.savefig('/home/aya/Streaming-Fraud-Detection/resultats/gender_vs_fraud.png')


# In this case, we do not see a clear difference between both genders. Data seem to suggest that females and males are almost equally susceptible (50%) to transaction fraud. Gender is not very indicative of a fraudulent transaction.

# ### 4. Spending Category vs Fraud
# 
# we examine in which spending categories fraud happens most predominantly. To do this, we first calculate the distribution in normal transactions and then the the distribution in fraudulent activities. The difference between the 2 distributions will demonstrate which category is most susceptible to fraud. For example, if 'grocery_pos' accounts for 50% of the total in normal transactions and 50% in fraudulent transactions, this doesn't mean that it is a major category for fraud, it simply means it is just a popular spending category in general. However, if the percentage is 10% in normal but 30% in fraudulent, then we know that there is a pattern.

# In[16]:


#calculate the percentage difference
a=df[df['is_fraud']==0]['category'].value_counts(normalize=True).to_frame().reset_index()
a.columns=['category','not fraud percentage']

b=df[df['is_fraud']==1]['category'].value_counts(normalize=True).to_frame().reset_index()
b.columns=['category','fraud percentage']

ab=a.merge(b,on='category')
ab['diff']=ab['fraud percentage']-ab['not fraud percentage']

unique_categories = ab['category'].unique()
palette = sns.color_palette("husl", len(unique_categories))

color_dict = dict(zip(unique_categories, palette))

plt.figure(figsize=(10, 6))
ax = sns.barplot(y='category', x='diff', data=ab.sort_values('diff', ascending=False), 
                 palette=[color_dict[cat] for cat in ab.sort_values('diff', ascending=False)['category']])

ax.set_xlabel('Percentage Difference')
ax.set_ylabel('Transaction Category')
plt.title('Percentage Difference of Fraudulent over Non-Fraudulent Transactions by Category')

plt.tight_layout()
plt.savefig('/home/aya/Streaming-Fraud-Detection/resultats/percentage_diff_fraud_vs_nonfraud.png')



# Some spending categories indeed see more fraud than others! Fraud tends to happen more often in 'Shopping_net', 'Grocery_pos', and 'misc_net' while 'home' and 'kids_pets' among others tend to see more normal transactions than fraudulent ones.

# ### 5.Age vs Fraud

# In[17]:


#age vs fraud
import datetime as dt
df['age']=dt.date.today().year-pd.to_datetime(df['dob']).dt.year
ax=sns.kdeplot(x='age',data=df, hue='is_fraud', common_norm=False)
ax.set_xlabel('Credit Card Holder Age')
ax.set_ylabel('Density')
plt.xticks(np.arange(0,110,5))
plt.title('Age Distribution in Fraudulent vs Non-Fraudulent Transactions')
plt.legend(title='Type', labels=['Fraud', 'Not Fraud'])
plt.savefig('/home/aya/Streaming-Fraud-Detection/resultats/age_distribution_vs_fraud.png')

# Ferme le graphique pour éviter les conflits avec d'autres graphiques
plt.close()

# The age distribution is visibly different between 2 transaction types. In normal transactions, there are 2 peaks at the age of 37-38 and 49-50, while in fraudulent transactions, the age distribution is a little smoother and the second peak does include a wider age group from 50-65. This does suggest that older people are potentially more prone to fraud.

# ### 6. Cyclicality of Credit Card Fraud
# How do fraudulent transactions distribute on the temporal spectrum? Is there an hourly, monthly, or seasonal trend? We can use the transaction time column to answer this question.

# In[18]:


#time in a day vs fraud
df['hour']=pd.to_datetime(df['trans_date_trans_time']).dt.hour
ax=sns.histplot(data=df, x="hour", hue="is_fraud", common_norm=False,stat='percent',multiple='dodge')
ax.set_ylabel('Percentage')
ax.set_xlabel('Time (Hour) in a Day')
plt.xticks(np.arange(0,24,1))
plt.legend(title='Type', labels=['Fraud', 'Not Fraud'])
# Sauvegarde du graphique
plt.savefig('/home/aya/Streaming-Fraud-Detection/resultats/time_in_a_day_vs_fraud.png')
plt.close()

# A very sharp contrast! While normal transactions distribute more or less equally throughout the day, fraudulent payments happen disproportionately around midnight when most people are asleep!

# In[19]:
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Exemple de données pour illustration
# df = pd.read_csv(...) # Assure-toi que df est chargé correctement avant

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

# Jour de la semaine vs fraude
df['day'] = df['trans_date_trans_time'].dt.dayofweek  # Convertir la date en jour de la semaine (0=Mon, 6=Sun)
ax = sns.histplot(data=df, x="day", hue="is_fraud", common_norm=False, stat='percent', multiple='dodge')

# Vérifie et configure correctement les ticks
ax.set_xticks(np.arange(0, 7))  # Assurer que les ticks sont de 0 à 6 (jours de la semaine)
ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])  # Les jours de la semaine
ax.set_ylabel('Percentage')
ax.set_xlabel('Day of Week')
plt.legend(title='Type', labels=['Fraud', 'Not Fraud'])
# Sauvegarde du graphique
plt.savefig('/home/aya/Streaming-Fraud-Detection/resultats/day_of_week_vs_fraud.png')
plt.close()


# Mois vs fraude
df['month'] = df['trans_date_trans_time'].dt.month
ax = sns.histplot(data=df, x="month", hue="is_fraud", common_norm=False, stat='percent', multiple='dodge')

ax.set_ylabel('Percentage')
ax.set_xlabel('Month')
plt.xticks(np.arange(1, 13, 1))  # Les mois de l'année
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(title='Type', labels=['Fraud', 'Not Fraud'])
plt.savefig('/home/aya/Streaming-Fraud-Detection/resultats/month_vs_fraud.png')
plt.close()
