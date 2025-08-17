from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('backend/model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form.get('review_text')
    if not review:
        return render_template('index.html', error="No review submitted.")
    
    pred = model.predict([review])[0]
    proba = model.predict_proba([review])[0]
    confidence = round(max(proba) * 100, 2)

    return render_template(
        'result.html',
        review=review,
        result=pred,
        confidence=confidence
    )

if __name__ == '__main__':
    app.run(debug=True)
