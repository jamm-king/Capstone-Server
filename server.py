import json
import os
import cv2

from flask import Flask, request, jsonify
import app

server = Flask(__name__)
angle_data = []


@server.route('/', methods=['POST'])
def root():
    global angle_data

    if request.method == 'POST':
        data = request.json
        with open('media/angle_data.json', 'w') as json_file:
            json.dump(data, json_file)
        angle_data = data
        print(data)
        return jsonify({'message': 'POST request received', 'data': data})


@server.route('/video', methods=['POST'])
def video():
    global angle_data

    if request.method == 'POST':
        data = request.get_data()
        save_path = 'media/'
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'video.mp4'), 'wb') as f:
            f.write(data)

        app.main(angle_data)

        return jsonify({'message': 'POST request received'})


server.run(host='0.0.0.0')
