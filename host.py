from flask import Flask, jsonify, request, render_template, redirect, url_for, make_response, send_from_directory, session, flash
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.secret_key = '$L$yCa$N$'
    # app.run(ssl_context='adhoc')
    app.run(host="0.0.0.0",port="505",ssl_context='adhoc')