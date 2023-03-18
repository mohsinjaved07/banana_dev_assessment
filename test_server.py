from flask import Flask
from flask_cors import CORS
import os
from test_onnx import test_onnx_func
import time


app = Flask(__name__)
CORS(app)


@app.route("/images")
def jpeg_image_path():
    l = []
    for file in os.listdir("./"):
        if file.endswith(".jpeg") or file.endswith(".png") or file.endswith(".jpg"):
            l.append(file)


    return l


@app.route("/test_image")
def prediction():
    timings = []
    for file in os.listdir("./"):
        if file.endswith(".jpeg") or file.endswith(".png") or file.endswith(".jpg"):
            st = time.time()
            test_onnx_func(file)
            et = time.time()
            elapsed_time = et - st
            timings.append(elapsed_time)


    return {
        "Message" : "Here are the timings",
        "timings" : timings
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0")
