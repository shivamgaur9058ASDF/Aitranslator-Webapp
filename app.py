from flask import Flask, render_template, request
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

# Select model
model_name = "Helsinki-NLP/opus-mt-en-hi"

# Load model
model = MarianMTModel.from_pretrained(model_name)

# Load tokenizer (FIXED)
tokenizer = MarianTokenizer.from_pretrained(model_name)


def translation(data):
    # Convert text into tokens (tensor)
    inputs = tokenizer(data, return_tensors="pt", padding=True)

    # Generate translation
    translated = model.generate(**inputs)

    # Decode tokens into readable text
    result = tokenizer.decode(translated[0], skip_special_tokens=True)

    return result


@app.route('/', methods=['GET', 'POST'])
def index():
    translated_text = ""

    if request.method == 'POST':
        data = request.form['data']
        translated_text = translation(data)

    return render_template('index.html', translated_text=translated_text)


if __name__ == '__main__':
    app.run(debug=True)