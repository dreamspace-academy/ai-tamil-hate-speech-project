## Resources


For the (manual) annotation task we hosted an instance of [doccano](https://github.com/doccano/doccano) on AWS. This is an open source tool which is light weight and very suitable for text classification annotation. The details of how setting up an own instance - either locally or hosting it on a virtual machine - can be found in the tool documentation. 

## Folder structure

data \
│   ├── doccano_annotated.csv - (labeled and unlabeled) data extracted from doccano \
│   └── doccano_annotated.csv.dvc - dvc tracking file \
├── doccano.db - sqlite database resulting from hosted doccano service \
└── doccano.db.dvc - dvc tracking file 


## Continuing the annotation/using already set up annotation project

The file doccano_annotated.csv can be imported into a new project, once an own doccano instance is created. It is also possible to import all the metadata and essentially continue the annotation as if the instance used during the challenge was never closed. This is what the file doccano.db is for. When a new doccano docker container is created, this file should be mounted (as a volume) instead of the empty /data folder. It is also possible to simply copy the file _into_ the container and replace the existing one.