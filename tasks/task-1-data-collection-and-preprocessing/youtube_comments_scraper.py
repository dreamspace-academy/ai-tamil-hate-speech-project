# 1.Create a virtualenv
# 2. pip install https://github.com/egbertbouman/youtube-comment-downloader/archive/master.zip
import os
import glob
import json
import pandas as pd

output_csv_file = "gender_videos_output.csv"
#youtube_comments_id = ['qqpLXOVo2S0','msJHZbDL6IA','9YQTwJ7D928','dFpNMWH5cYA','PZiO8FHC0Hc']
youtube_comments_id = ['nGym0d6Kt44','wJ_JLjGS6m4','Hi9fs_GxVxU']
df_list = []
for link in youtube_comments_id:
    os.system(f'youtube-comment-downloader --youtubeid {link} --output {link}.txt')
json_files = glob.glob("*.txt")
for each_file in json_files:
    with open(each_file) as file:
        for line in file:
            j_content = json.loads(line)
            df_list.append(pd.DataFrame({'text': j_content['text'], 'time': j_content['time']},index=[0]))
df = pd.concat(df_list)
df.to_csv(output_csv_file)
