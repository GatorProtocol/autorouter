
from flask import Flask, request, jsonify

from router import Router
from runtime import run_app, run_trainer

import threading

router = Router()

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    query = request.json["query"]
    model_id = router.predict(query)

    return jsonify({"model": int(model_id)})

@app.route("/add", methods=["POST"])
def retrain():
    data = request.json
    model = data.get("model")
    query = data.get("query")

    router.add(query, model)


app_thread = threading.Thread(target=run_app, args=(app,))
trainer_thread = threading.Thread(target=run_trainer)

app_thread.start()
trainer_thread.start()