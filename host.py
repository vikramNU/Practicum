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

@app.route('/proc', methods=['GET','POST'])
def proc():
    print(1)
    data = request.get_json()
    print(data)
    return jsonify({'ans':data})

@app.route('/pred', methods=['GET','POST'])
def pred():
    #tags = json.loads(request.form['basic'])
    if request.method == "GET":
        #products = request.form.getlist('basic')
        #print("Products",products)
        print(request)
        req_url = str(request.url).split('?')[1].replace('%7B','').replace('%5B','').replace('%22','').replace('%7B','').replace('%20',' ').replace('%7D','').replace('%5D','')
        print("ORIG",req_url)
        req_lst = req_url.replace('value:','').split(',')
        print(req_lst)

    #print(tags)
    #return render_template('index.html')
    return jsonify(req_lst)

if __name__ == '__main__':
    app.secret_key = '$L$yCa$N$'
    app.run(port="1211",ssl_context='adhoc')
    # app.run(host="0.0.0.0",port="443",ssl_context='adhoc')
    
