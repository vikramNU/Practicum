from flask import Flask, jsonify, request, render_template, redirect, url_for, make_response, send_from_directory, session, flash
from infer import predict
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def oo():
    data = request.form['name']
    print(data)
    return render_template('index.html')

@app.route('/proc', methods=['GET,''POST'])
def pred():
    print(1)
    data = request.get_json()
    print(data)
    return jsonify({'ans':data})

if __name__ == '__main__':
    app.secret_key = '$L$yCa$N$'
    app.run(port="1211",ssl_context='adhoc')
    # app.run(host="0.0.0.0",port="443",ssl_context='adhoc')
    