import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy

# Load the dataset
file_path = r'C:\Users\0042H8744\movie_dataset.csv'
df = pd.read_csv(file_path)

# Drop rows with missing values in the 'vote_average' or 'vote_count' columns
df = df.dropna(subset=['vote_average', 'vote_count'])

# Convert 'genres' column to a list of strings
df['genres'] = df['genres'].apply(lambda x: str(x).split(',') if pd.notna(x) else [])

# Extract the genre names from the list of strings
df['genres'] = df['genres'].apply(lambda x: [genre.strip() for genre in x])

# Create a Surprise Reader object
reader = Reader(rating_scale=(0, 10))

# Create a dummy user column with constant value
df['user'] = 1

# Load the data into the Surprise format
data = Dataset.load_from_df(df[['user', 'id', 'vote_average']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Choose the algorithm (SVD - Singular Value Decomposition)
model = SVD(n_factors=100, lr_all=0.005, reg_all=0.02)  # Example hyperparameters

# Train the model on the training set
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")

# Example: Get movie recommendations for a specific user
user_id = 1  # Replace with the desired user ID
user_ratings = df[df['id'] == user_id][['id', 'vote_average']]
user_ratings.columns = ['id', 'rating']
user_ratings = user_ratings.dropna()

# Generate a list of movies the user has not rated
movies_to_recommend = df[~df['id'].isin(user_ratings['id'])]['id'].unique()

# Predict ratings for movies not rated by the user
user_predictions = [model.predict(user_id, movie_id) for movie_id in movies_to_recommend]

# Get top N recommended movies
N = 5
top_movies = sorted(user_predictions, key=lambda x: x.est, reverse=True)[:N]

# Display the top recommended movies
for movie in top_movies:
    print(f"Movie ID: {movie.iid}, Estimated Rating: {movie.est}")
