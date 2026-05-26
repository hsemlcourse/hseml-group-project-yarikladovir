import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")


st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🕵️",
    layout="centered",
)

st.title("Fake Job Posting Detector")
st.write(
    "This demo uses a TF-IDF + Linear SVC model to detect potentially fraudulent job postings."
)

example_text = (
    "Earn money from home. No experience required. "
    "Send your bank details for registration."
)

text = st.text_area(
    "Job posting text",
    value=example_text,
    height=200,
)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter a job posting text.")
    else:
        try:
            response = requests.post(
                API_URL,
                json={"text": text},
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()

            prediction = result["prediction"]
            is_fraud = result["is_fraud"]
            score = result["score"]
            model_name = result["model_name"]

            if is_fraud is True:
                st.error("Prediction: potentially fraudulent job posting")
            elif is_fraud is False:
                st.success("Prediction: likely real job posting")
            else:
                st.info(f"Raw prediction: {prediction}")

            st.write("Model:", model_name)
            st.write("Raw prediction:", prediction)

            if score is not None:
                st.write("Decision score:", round(score, 4))

            st.json(result)

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the FastAPI server. "
                "Start it with: uvicorn src.api:app --reload"
            )
        except requests.exceptions.RequestException as error:
            st.error(f"Request failed: {error}")