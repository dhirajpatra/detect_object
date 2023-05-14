from flask import Flask, request, jsonify, render_template
import argparse
import sys
import time
import numpy as np
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils


WIDTH = 600
HEIGHT = 400
IMAGE_SIZE = (WIDTH, HEIGHT)
app = Flask(__name__)


@app.route("/")
def index():
    # Render the template file.
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    # Get the image file from the request body.
    image = request.files["image"]

    # Convert the byte string into a NumPy array using OpenCV.
    image_array = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Resize the image to 200x200 pixels.
    image = cv2.resize(image_array, IMAGE_SIZE)

    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Initialize the object detection model.
    base_options = core.BaseOptions(
        file_name="efficientdet_lite0.tflite", use_coral=False, num_threads=4
    )
    detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options
    )
    detector = vision.ObjectDetector.create_from_options(options)

    # Run object detection on the image.
    detection_result = detector.detect(input_tensor)

    # Draw keypoints and edges on input image
    image = utils.visualize(image, detection_result)

    # Stop the program if the ESC key is pressed.
    while cv2.waitKey(1) != 27:
        cv2.resizeWindow("object_detector", WIDTH, HEIGHT)
        cv2.imshow("object_detector", image)

    cv2.destroyAllWindows()

    objects = []
    for detection in detection_result.detections:
        for category in detection.categories:
            objects.append(
                {"category_name": category.category_name, "score": category.score}
            )

    # Convert the detection result to a JSON object.
    results = {"objects": objects}

    # Return the JSON object.
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
