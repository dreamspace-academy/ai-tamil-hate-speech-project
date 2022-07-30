# Importing packages: streamlit for the frontend, requests to make the api calls
import streamlit as st
import requests
import json


class MakeCalls:
    def __init__(self, url: str = "http://localhost:8080/") -> None:
        """
        Constructor for the MakeCalls class. This class is used to perform API calls to the backend service.
        :param url: URL of the server. Default value is set to local host: http://localhost:8080
        """
        self.url = url
        self.headers = {"Content-Type": "application/json"}

    def model_list(self, service: str) -> dict:
        """
        add doctext here.
        """
        model_info_url = self.url + f"api/v1/{service}/info"
        models = requests.get(url=model_info_url)
        return json.loads(models.text)

    def run_inference(
        self, service: str, model: str, text: str, query: str = None
    ) -> json:
        """
        add doctext here.
        """
        inference_enpoint = self.url + f"api/v1/{service}/predict"

        payload = {"model": model.lower(), "text": text, "query": query.lower()}
        result = requests.post(
            url=inference_enpoint, headers=self.headers, data=json.dumps(payload)
        )
        return json.loads(result.text)


class Display:
    def __init__(self):
        st.title("Omdena NYU")
        st.sidebar.header("NYU Omdena Hate Speech Classification")
        self.service_options = st.sidebar.selectbox(
            label="",
            options=[
                "Hate Classification",
            ],
        )
        self.service = {
            "Hate Classification": "classification",
        }

    def static_elements(self):
        return self.service[self.service_options]

    def dynamic_element(self, models_dict: dict):
        """
       add doc text here.
        """
        
        st.header(self.service_options)
        model_name = list()
        model_info = list()
        for i in models_dict.keys():
            model_name.append(models_dict[i]["name"])
            model_info.append(models_dict[i]["info"])
        st.sidebar.header("Model Information")
        #for i in range(len(model_name)):
            #st.sidebar.subheader(model_name[i])
            #st.sidebar.info(model_info[i])
        model: str = st.selectbox("Select the Trained Model", model_name)
        input_text: str = st.text_area("Enter Text here")
        query: str = "None"
        run_button: bool = st.button("Run")
        return model, input_text, query, run_button


def main():

    page = Display()
    service = page.static_elements()
    apicall = MakeCalls()
    model_details = apicall.model_list(service=service)
    model, input_text, query, run_button = page.dynamic_element(model_details)
    if run_button:
            with st.spinner(text="Getting Results.."):
                result = apicall.run_inference(
                    service=service,
                    model=model.lower(),
                    text=input_text,
                    query=query.lower(),
                )
            st.write(result)


if __name__ == "__main__":
    main()
