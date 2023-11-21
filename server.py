import os

from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/', methods=['POST'])
def root():
    if request.method == 'POST':
        data = request.json
        print(data)
        return jsonify({'message': 'POST request received', 'data': data})


@app.route('/video', methods=['POST'])
def video():
    if request.method == 'POST':
        # Save the binary data to a file
        data = request.get_data()
        save_path = 'media/'  # Change this to the desired directory
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'video.3gp'), 'wb') as f:
            f.write(data)
        return jsonify({'message': 'POST request received'})


app.run(host='0.0.0.0')
