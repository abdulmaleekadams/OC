from flask import Flask
from flask_cors import CORS
from routes import main_bp

app = Flask(__name__)

# Register the blueprint
app.register_blueprint(main_bp)

# Apply CORS to the app
CORS(app, origins=['http://localhost:3000', 'https://oc-predictor.vercel.app'])

if __name__ == '__main__':
    app.run()
