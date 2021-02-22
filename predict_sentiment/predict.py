import json
import numpy as np
from flask import Flask, jsonify, request
import requests
from prepare_data import PrepareData
import config

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify("API is up and running. Use the predict end point.")

def _infer(model_input):
    r = requests.post(config.MODEL_URL, json=model_input)
    predictions = json.loads(r.text.encode("utf-8"))
    return predictions

@app.route("/predict", methods=["POST"])
def predict_sentiment():
    input_json = request.get_json()
    input_review = json.loads(input_json["review"])
    prep = PrepareData(input_fields=config.INPUT_FIELDS,
                        text_field=config.TEXT_FIELD,
                        target_field=config.TARGET_FIELD,
                        maxlen=config.MAX_LEN,
                        padding="pre",
                        vocab_size=config.VOCAB_SIZE)
    x = prep.run_prep(input_review)
    predicted_prob = _infer(model_input={"inputs": {"text_input": x.values.tolist()}})["outputs"]
    predicted_prob = np.array(predicted_prob)
    sentiments = np.where(predicted_prob > 0.5, 1, 0)    
    return jsonify(sentiments.tolist())


if __name__ == "__main__":
    app.run(port=5000)