# 1.Create a virtualenv
# 2. pip install https://github.com/egbertbouman/youtube-comment-downloader/archive/master.zip
import os
import glob
import json
import pandas as pd
from googletrans import Translator
#output_csv_file = "gender_videos_output.csv"
output_csv_file = "political_videos_output.csv"
youtube_comments_id = ['qqpLXOVo2S0','msJHZbDL6IA','9YQTwJ7D928','dFpNMWH5cYA','PZiO8FHC0Hc']
#Gender Videos
#youtube_comments_id = ['nGym0d6Kt44','wJ_JLjGS6m4','Hi9fs_GxVxU','GdJaZclrAR0','vKMz4iUfCGs','o9DON6PnF4M']
#Religion Videos
#youtube_comments_id = ['h5ePiMiuiYA','ho96m3erCdw']
translator = Translator()
#output_csv_file = "test.csv"
#youtube_comments_id = ['msJHZbDL6IA']
#trans = translator.detect('이 문장은 한글로 쓰여졌습니다.')
#trans = translator.detect("Tamilan")
df_list = []
for link in youtube_comments_id:
    os.system(f'youtube-comment-downloader --youtubeid {link} --output {link}.txt')
json_files = glob.glob("*.txt")
for each_file in json_files:
    with open(each_file) as file:
        for line in file:
            j_content = json.loads(line)
            google_trans = translator.detect(j_content["text"])
            if type(google_trans.lang) == list:
                if not('en' in google_trans.lang and google_trans.confidence[google_trans.lang.index('en')]>=0.5):
                    df_list.append(pd.DataFrame({'text': j_content['text'], 'time': j_content['time']},index=[0]))
            else:
                if not(google_trans.lang == 'en' and google_trans.confidence>=0.5):
                    df_list.append(pd.DataFrame({'text': j_content['text'], 'time': j_content['time']},index=[0]))
df = pd.concat(df_list)
os.system("rm -rf *.txt")
df.to_csv(output_csv_file)
'''
google_trans = translator.detect("  @Dhana Sowndar  proud to say that I'm tamilan...")
print(google_trans)
'''