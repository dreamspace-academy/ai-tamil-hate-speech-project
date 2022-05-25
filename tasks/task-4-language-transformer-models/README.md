# Resources
A summary of resources for the hate speech detection task. Feel free to add
resources that can be helpful for the task.

## Datasets
#### Public datasets
[**DravidianCodeMix-Dataset**](https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset)

The dataset was annotated for sentiment analysis and offensive language
identification for a total of more than 60,000 YouTube comments. The dataset
consists of around 44,000 comments in Tamil-English, around 7,000 comments in
Kannada-English, and around 20,000 comments in Malayalam-English.
 - [paper](https://arxiv.org/pdf/2106.09460.pdf)

[**IndicCorp**](https://indicnlp.ai4bharat.org/corpora/)

One of the largest publicly-available corpora for Indian languages

[**tamilmixsentiment**](https://huggingface.co/datasets/tamilmixsentiment)

Tamil-English code-switched sentiment-annotated corpus containing 15,744 comment
posts from YouTube


#### Omdena datasets

**Under development**

[YouTube Comments](https://dagshub.com/Omdena/NYU/src/master/tasks/task-1-data-collection-and-preprocessing/youtube_data)

[Twitter Comments](https://drive.google.com/drive/folders/1UERgoC5EnVFbQAQzdcnScDaf-iL5LwUF)

[dataset in google Drive](https://drive.google.com/drive/folders/12ACbgg0SVIdYuKYsocp_FRIVxBxZZru_)


## Models and libraries

**indicBERT**

[**indicBERT**](https://huggingface.co/ai4bharat/indic-bert) is a multilingual
ALBERT model pretrained on 12 major India languages including Assamese, Bengali,
English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Oriya, Punjabi, Tamil and
Telugu.
 - [paper](https://indicnlp.ai4bharat.org/papers/arxiv2020_indicnlp_corpus.pdf)


**Dravidian-Offensive-Language-Identification**

Shared task Offensive Language Identification in Dravidian Languages at
Dravidian Language Technology Workshop at EACL 2021.
 - [paper](https://arxiv.org/pdf/2102.07150.pdf)
 - [github](https://github.com/kushal2000/Dravidian-Offensive-Language-Identification)

**TaMillion**

[**TaMillion**](https://huggingface.co/monsoon-nlp/tamillion) is the second
version of a Tamil language model trained with Google Research's ELECTRA. The
model is trained on [IndicCorp Tamil](https://indicnlp.ai4bharat.org/corpora/)
(11GB) and 1 October 2020 dump of [tamil-wikipedia](https://ta.wikipedia.org)
(482MB)

**GPT2-Tamil**

[**GPT2-Tamil**](https://huggingface.co/abinayam/gpt-2-tamil) is a pretrained
model on Tamil language([oscar dataset](https://huggingface.co/datasets/oscar)
and [IndicNLP dataset-ta](https://indicnlp.ai4bharat.org/corpora/)) using a causal
language modeling(CLM) objective.

**deoffxlmr-mono-tamil**

[**deoffxlmr-mono-tamil**](https://huggingface.co/Hate-speech-CNERG/deoffxlmr-mono-tamil)
is a model used to detect Offensive Content in Tamil Code-Mixed language.
 - [paper](https://aclanthology.org/2021.dravidianlangtech-1.38/)
 - [github](https://github.com/hate-alert/Hate-Alert-DravidianLangTech)

**XLM-RoBERTa-base-ft-udpos28-ta**

[**XLM-RoBERTa-base-ft-udpos28-ta**](https://huggingface.co/wietsedv/xlm-roberta-base-ft-udpos28-ta) This model is part of the project cross-lingual part-of-speech tagging.
 - [project](https://huggingface.co/spaces/wietsedv/xpos)
 - [paper](https://wietsedv.nl/files/devries_acl2022.pdf)

## Tools



## Reading materials

 - [Transformer paper](https://arxiv.org/pdf/1706.03762.pdf)
 - [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
 - [BERT paper](https://arxiv.org/pdf/1810.04805.pdf)
 - [Towards generalisable hate speech detection: a review on obstacles and solutions](https://peerj.com/articles/cs-598/)




## Other resources
[**Tamil Deep Learning Awesome List**](https://narvidhai.github.io/tamil-nlp-catalog/#/)

[**IndicNLP**](https://indicnlp.ai4bharat.org/home/) -- models and resources for
Indian languages

[**NLP for Code-mixed Tamil-English**](https://github.com/goru001/nlp-for-tanglish)

[**Huggingface Trabsformer Course**](https://huggingface.co/course/chapter0/1)

[**FastAI-Practical Deep Learning for Coders**](https://course.fast.ai/)

[**How to Finetune BERT for Text Classification (HuggingFace Transformers, Tensorflow 2.0) on a Custom Dataset**](https://victordibia.com/blog/text-classification-hf-tf2/)

