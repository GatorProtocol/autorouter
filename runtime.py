from router import Router

import time

def run_app(app):
    app.run(host="0.0.0.0", port=80)

def run_trainer():
    router = Router()
    time.sleep(600)
    router.retrain()