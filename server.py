from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/', methods=['POST'])
def root():
    if request.method == 'POST':
        data = request.json
        print(data)
        return jsonify({'message': 'POST request received', 'data': data})


app.run(debug=True)
