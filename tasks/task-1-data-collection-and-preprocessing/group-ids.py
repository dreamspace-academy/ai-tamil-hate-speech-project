links = [
'https://www.facebook.com/lellahutanpage/',
'https://www.facebook.com/Gangsterrathne/',
'https://www.facebook.com/buwaSL/',
'https://www.facebook.com/sampathathukoralaofficialpage/',
'https://www.facebook.com/sinhalesinhayo12/',
'https://www.facebook.com/TheRealMadaawiya/',
'https://www.facebook.com/ApiSriLankanBro/',
'https://www.facebook.com/SriLankanBlood1/',
'https://www.facebook.com/apemedia.lk/',
'https://www.facebook.com/apearata/',
'https://www.facebook.com/sinhalayaone/',
'https://www.facebook.com/SinhalaSangedama/',
'https://www.facebook.com/WeR4SL1/',
'https://www.facebook.com/SinhalaRavaya/',
'https://www.facebook.com/AllSriLankanMuslims/',
'https://www.facebook.com/Dasun.M.W',
'https://www.facebook.com/maraya001original',
'https://www.facebook.com/noaddress.lk',
'https://www.facebook.com/AzzamAmeenSL/',
'https://www.facebook.com/groundviews/',
'https://www.facebook.com/indrajithbanda2/',
'https://www.facebook.com/soraya.deen.3/',
'https://www.facebook.com/TamilGuardian/',
'https://www.facebook.com/tamilwinnews/',
'https://www.facebook.com/muslimulamaparty/',
'https://www.facebook.com/MamaSinhalaEiAulda/',
'https://www.facebook.com/Muzammil/',
'https://www.facebook.com/Jamiyya/',
'https://www.facebook.com/srilankamuslim/',
'https://www.facebook.com/sltraditionalmuslims/',
'https://www.facebook.com/Venathuraliyerathana/'
]

grps = []

for i in links:
    i = i.split('/')
    grps.append(i[-2])

print(grps)