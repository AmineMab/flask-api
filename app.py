from flask import Flask, jsonify, request, render_template
from transformers import BertForSequenceClassification, BertTokenizer
import torch

app = Flask(__name__)

# Load the BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-cased")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Set the model to evaluation mode
model.eval()

@app.route('/', methods=['GET'])
def home():
    # Render the HTML template for the homepage
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text to classify from the request body
    text = request.json['text']

    # Tokenize the text
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)

    # Pass the input through the model
    output = model(input_ids)

    # Get the logits
    logits = output[0]

    # Get the index of the highest logit
    index = torch.argmax(logits)

    # Get the label corresponding to the highest logit
    label = model.config.id2label[index.item()]

    # Create a response to return to the client
    response = {
        'label': label
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=8000)