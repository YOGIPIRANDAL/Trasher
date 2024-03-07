# app.py
import os

from flask import Flask
from flask_cors import CORS

from routes import register_blueprints

app = Flask(__name__)
CORS(app)

# Register blueprints
register_blueprints(app)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))