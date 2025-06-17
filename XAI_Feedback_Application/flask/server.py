from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, supports_credentials=True)


if "APPSTORE_ENV" in os.environ:
    from myProxyFix import ReverseProxied

    app.wsgi_app = ReverseProxied(app.wsgi_app)


@app.route("/")
def hello():
    # input = request.json['input']
    return jsonify(dict(output="Hello World"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")