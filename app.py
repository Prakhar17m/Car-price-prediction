from flask import Flask, render_template
from models.model_utils import run_all_models

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    metrics, plot_paths = run_all_models()
    return render_template('results.html', metrics=metrics, plots=plot_paths)

if __name__ == '__main__':
    app.run(debug=True)
