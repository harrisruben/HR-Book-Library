from flask import Flask, render_template, request, jsonify
import pandas as pd
from model import load_data, train_model, recommend_books

app = Flask(__name__)

user_item_matrix = load_data()
model = train_model(user_item_matrix)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    recommendations = recommend_books(model, user_id, user_item_matrix)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
