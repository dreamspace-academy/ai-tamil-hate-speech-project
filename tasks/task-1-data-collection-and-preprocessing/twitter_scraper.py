#twitter scraper and preprocessor

import pandas as pd
import numpy as np
import os

# keywords

KEYWORD = keyword  # insert keyword

os.system("snscrape --jsonl --progress --max-results 500 twitter-search KEYWORD > new_tweets.json")

df_new = pd.read_json('new_tweets.json', lines=True)

# import all_tweets
'''
this won't exist yet if this is 1st time running script. 
if that's the case, comment out all lines related to df_all,
and export df_new as all_tweets.json.
then once we have that table, we can use this script as it is to buiold on all_tweets.json table.
'''

df_all = pd.read_json('all_tweets.json')

# FEATURE ENGINEERING/SELECTION before concatenating & exporting

df_new = df_new[['date', 'id', 'user', 'content', 'hashtags', 'mentionedUsers']]

# extract features from users col
usernames = list()
display_names = list()
mentioned_usernames = list()

for user in df_new['user']:
    usernames.append(user['username'])
    display_names.append(user['displayname'])

'''
# FIXME
df_new['mentionedUsers'].fillna(0, inplace=True)
for user in df_new['mentionedUsers']:
    mentioned_usernames.append(user['username'])
'''

df_new['username'] = usernames
df_new['display_name'] = display_names
# df_new['mentioned_usernames'] = mentioned_usernames

'''
# extracting mentioned usernames from mentionedUsers
df_new['mentionedUsers'].fillna(0, inplace=True)
mentioned_usernames = list()
for i in range(len(df_new)):
    mentions = list()
    for mention in df_new['mentionedUsers'][i]:
        if mention == 0:  #FIXME
            mentions.append('none')
        else:
            mentions.append(mention['username'])
    mentioned_usernames.append(mentions)
    # print(mentions)
'''

df_new['hashtags'] = df_new['hashtags'].fillna('none')
# df_new['mentionedUsers'] = df_new['mentionedUsers'].fillna('none')
df_new['id'] = df_new['id'].astype('str')
df_new['date'] = df_new['date'].apply(lambda a: pd.to_datetime(a).date())

# Append df_new to df_all
df_all = df_all.append(df_new).reset_index(drop=True)

# drop duplicate rows based on a unique identifier column ('id')
# len(df_all), len(df_all['id'].unique())
df_all.drop_duplicates(subset=['id'], inplace=True)
df_all.reset_index(drop=True, inplace=True)

# export df_all as json
df_all.to_json('all_tweets.json')
