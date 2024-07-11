import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Load the dataset
df = pd.read_csv('tmdb_5000_movies.csv')

# Parse the first row of genres
X = df.iloc[0]

# Fixing the parsing of JSON in genres
j = json.loads(X['genres'])

# Combine genre names
' '.join(''.join(jj['name'].split()) for jj in j)

# Function to combine genres and keywords into a single string
def genres_and_keywords_to_string(row):
    genres = json.loads(row['genres'])
    genres = ' '.join(''.join(j['name'].split()) for j in genres)
    keywords = json.loads(row['keywords'])
    keywords = ' '.join(''.join(j['name'].split()) for j in keywords)
    return f"{genres} {keywords}"

# Apply the function to the dataframe
df['string'] = df.apply(genres_and_keywords_to_string, axis=1)

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=2000)

# Fit and transform the 'string' column
X = tfidf.fit_transform(df['string'])

# Create a Series mapping movie titles to their index
movie2idx = pd.Series(df.index, index=df['title'])

# Get the index for a specific movie
idx = movie2idx['Scream 3']

# Get the TF-IDF vector for the queried movie
query = X[idx]

# Calculate cosine similarity
scores = cosine_similarity(query, X)
print(scores)

# Flatten the scores
scores = scores.flatten()

# Plot the scores
plt.plot(scores)

# Sort and plot the sorted scores
plt.plot(scores[(-scores).argsort()])

recommended_idx = (-scores).argsort()[1:6]
df['title'].iloc[recommended_idx]
# Display the plots
plt.show()

def recommend(title):
    idx = movie2idx[title]
    if type(idx)==pd.Series:
        idx = idx.iloc[0]
    
    query = X[idx]
    scores=cosine_similarity(query,X)
    scores = scores.flatten()

    recommended_idx = (-scores).argsort()[1:6]
    return df['title'].iloc[recommended_idx]

print('Recommendations for "Scream 3": ')
print(recommend('Scream 3'))

print('Recommendations for "Mortal Kombat": ')
print(recommend('Mortal Kombat'))

print('Recommendations for "Runaway Bride": ')
print(recommend('Runaway Bride'))

print('Recommendations for "Spider-Man": ')
print(recommend('Spider-Man'))