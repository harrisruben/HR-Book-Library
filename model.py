import pandas as pd
from sklearn.decomposition import TruncatedSVD
def load_data(file_path="data/ratings.csv"):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()  
    if 'book_id' not in data.columns:
        raise ValueError("The 'book_id' column is missing from the dataset.")
    print("Columns:", data.columns)
    print(data.head())
    user_item_matrix = data.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
    return user_item_matrix
def train_model(user_item_matrix):
    svd = TruncatedSVD(n_components=4)
    svd.fit(user_item_matrix)
    return svd
def recommend_books(model, user_id, user_item_matrix, n_recommendations=5):
    user_vector = model.transform([user_item_matrix.loc[user_id].values])[0]
    item_scores = model.components_.T @ user_vector
    recommended_items = user_item_matrix.columns[item_scores.argsort()[-n_recommendations:][::-1]]
    return recommended_items.tolist()