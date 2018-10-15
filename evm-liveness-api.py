from flask import request
from flask import json
from flask import Flask
from flask import jsonify
import base64
import EVM

app = Flask(__name__)

@app.route('/liveness', methods = ['POST'])
def api_message():

    if request.headers['Content-Type'] == 'application/octet-stream':
        f = open('./output.avi', 'wb')
        # decoded_string = base64.b64decode(request.data)
        f.write(request.data)
        f.close()
        
        livenessValue = EVM.findLiveness()
        print (livenessValue)
        #return jsonify("Liveness", str(livenessValue))
        return livenessValue

    else:
        return "415 Unsupported Media Type"

if __name__ == '__main__':
    app.run()