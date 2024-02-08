from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved pipeline and label encoder
pipeline = joblib.load('your_pipeline.joblib')
label_encoder = joblib.load('label_encoder.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        graduation = request.form['graduation']
        interest = request.form['interest']
        prerequisites = request.form['prerequisites']
        entrance_test_score = float(request.form['entrance_test_score'])

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'Graduation': [graduation],
            'Interest': [interest],
            'prerequisites': [prerequisites],
            'entrance test score': [entrance_test_score]
        })

        # Use the label encoder to transform the input data
        input_data['Recommend Course'] = label_encoder.inverse_transform(pipeline.predict(input_data))

        return render_template('index.html', recommended_course=input_data['Recommend Course'].values[0])

if __name__ == '__main__':
    app.run(debug=True)

