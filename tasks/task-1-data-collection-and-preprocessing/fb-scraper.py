from facebook_scraper import get_posts
import pandas as pd

df = pd.DataFrame(columns = ['Group Name', 'Articles', 'HS Words'])



groups = ['lellahutanpage', 'Gangsterrathne', 'buwaSL', 'sampathathukoralaofficialpage', 'sinhalesinhayo12', 'TheRealMadaawiya', 'ApiSriLankanBro', 
'SriLankanBlood1', 'apemedia.lk', 'apearata', 'sinhalayaone', 'SinhalaSangedama', 'WeR4SL1', 'SinhalaRavaya', 'AllSriLankanMuslims', 'www.facebook.com', 'www.facebook.com', 'www.facebook.com', 'AzzamAmeenSL', 'groundviews', 'indrajithbanda2', 'soraya.deen.3', 'TamilGuardian', 'tamilwinnews', 'muslimulamaparty', 'MamaSinhalaEiAulda', 'Muzammil', 'Jamiyya', 'srilankamuslim', 'sltraditionalmuslims', 'Venathuraliyerathana']

hs = ['පොන්නයා','හම්බයො','මුස්ලිම් හැත්ත','තම්බි','මිනීමරුවො','වේසියො','ගණිකාවො','ගොන් බඩු','ත්‍රස්ථවාදියො','පර දෙමලු',
'කල්ලතෝනියො','தேவடியாள்','வேச, வேசி','சோனகன், சோனி','தொப்பி பிரட்டி','மொட்டை','மத வெறியர்',
'இனத்துவேசி','மோட்டு சிங்களவன்','பயங்கரவாதி/ தீவிரவாதி','அல்லேலூயா குறூப்','தோட்டக் காட்டான்',
'வந்தேறி','புர்க்கா','அயிட்டம் /துன்டு/ பிகர்','மட்டக்களப்பார்','யாழ்ப்பாணிகள்','யாழ்ப்பாணி','யாழ்ப்பானி']

for i in groups:
    print(f"Group: {i}")
    try:
        for post in get_posts(i, pages=30, options={"posts_per_page": 50}):
            if any(x in post['text'] for x in hs):
                df = df.append({'Group Name': i, 'Articles': post['text'], 'HS Words':1},ignore_index=True)
            else:
                df = df.append({'Group Name': i, 'Articles': post['text'], 'HS Words':0},ignore_index=True)
    except Exception as e:
        print(f"Error: {e}\nGroup : {i}")
        continue


df.to_csv("fb_scraped_date.csv")