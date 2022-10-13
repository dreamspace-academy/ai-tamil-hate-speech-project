import torch
from transformers import AutoTokenizer
from app.api.bert_model_artifacts.network import SentClf

device = torch.device("cpu")

class ClassProcessor:
    def __init__(self, model_name: str = None, service: str = "classification"):
        """
        Constructor to the class that does the Classification in the back end
        :param model: Transfomer model that will be used for Classification Task
        :param service: string to represent the service, this will be defaulted to classification
        """
        if model_name is None:
            model_name = "bert_model_artifacts"
            
        # path to model artifacts
        self.path = f"./app/api/{model_name}/"
        
        # args and weights for loading fine-tuned model
        self.args_path = self.path + "args.pt"
        self.model_path = self.path + "model_best.pt"
        
        # Selecting the correct model based on the passed madel input
        self.args = torch.load(self.args_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Loading model in {self.device} device")
        self.tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased", use_fast=True)
        self.model = SentClf(self.args)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

    def tokenize(self, input_text: str):
        """
        Method to tokenize the textual input
        :param input_text: Input text
        :return: Returns encoded text for inference
        """
        model_inputs = self.tokenizer(
		    input_text,
		    truncation=True,
		    max_length=512,
		    return_tensors='pt'
		).to(self.device)

        return model_inputs


    def inference(self, input_text: str):
        """
        Method to perform the inference
        :param input_text: Input text for the inference
        :return: correct category and confidence for that category
        """
        tokenized_inputs = self.tokenize(input_text)
        
        # Model returns the raw logtits
        logits = self.model(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'])
        
        # Transform logits into actual probabilities
        prob = torch.sigmoid(logits).cpu().detach().numpy()[0]
        
        # A single probability is returned. Threshold the probability into ether 0 or 1
        pred_index = 1 if prob[1]>0.4 else 0
        
        # Get label associated to index
        pred_label = self.args.labels[pred_index]
        
        # Calculate confidence
        confidence = prob[pred_index]
        
        return pred_label, confidence
