from flask_cors import CORS
from flask import Flask, request, render_template, json, jsonify, send_from_directory
import json
import cv2
import numpy as np
import io
from base64 import encodebytes, b64encode
from PIL import Image
from flask import make_response, jsonify

app = Flask(__name__)

CORS(app)


@app.route("/", methods=["GET"])
def main():
    return render_template('index.html')


@app.route("/api/prepare", methods=["POST"])
def prepare():
    res = {}
    file = request.files["file"]
    color = request.form.get('color')
    imagenes = preprocessing(file)
    if imagenes:
        for i, imagen in enumerate(imagenes):
            res[i] =[imagen[0].tolist(), imagen[1], color] #el tensor de imagen y el arrayBytes de imagen.
        return json.dumps(res)
    else:
        return make_response(jsonify("hay errores"), 404)


@app.route('/model')
def model():
    json_data = json.load(open("./model_js/model.json"))
    return jsonify(json_data)


@app.route('/<path:path>')
def load_shards(path):
    return send_from_directory('model_js', path)


def preprocessing(file):
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    #data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    #img = cv2.imdecode(data, 0)
    #res = cv2.resize(img, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
    # file.save("static/UPLOAD/img.png") # saving uploaded img
    # cv2.imwrite("static/UPLOAD/test.png", res) # saving processed image
    
    #return res
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    npimg = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    # convert numpy array to image
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    #image = cv2.imread('./Imagenes/s001-04_img.tif')
    imageAux = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, 1.1,5) #este valor puede varias mas o menos de 5 a 12 dependiendo de la calidad de las imagenes

    count = 0
    res = []
    for (x,y,w,h) in faces:
        rostro = imageAux[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(48,48), interpolation=cv2.INTER_CUBIC) #definir que tamaño van a tener
        rostroN = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
        pil_img = Image.fromarray(rostroN) # reads the PIL image
        byte_arr = io.BytesIO()
        pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
        encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
        res.append([rostroN, encoded_img])  
    return res


if __name__ == "__main__":
    app.run()
