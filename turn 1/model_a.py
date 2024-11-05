import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

# Example data
data = {
    'Site': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    'Mineral_1': [1, 0, 1, 0, 1, 1, 0],
    'Mineral_2': [0, 1, 1, 1, 0, 0, 1],
    'Mineral_3': [1, 0, 0, 1, 1, 0, 1],
    'Mineral_4': [0, 1, 0, 0, 1, 1, 0],
    'Geological_Feature_1': [0.7, 0.4, 0.8, 0.6, 0.3, 0.9, 0.5],
    'Geological_Feature_2': [0.2, 0.6, 0.3, 0.7, 0.8, 0.1, 0.4],
    'Geological_Feature_3': [0.5, 0.8, 0.6, 0.2, 0.4, 0.7, 0.3]
}
df = pd.DataFrame(data)


def collaborative_filtering(data, user_id, similarity_measure='cosine', k=3):
    # Convert the data to a user-item matrix
    user_item_matrix = data.drop(columns=['Site']).transpose()
    user_item_matrix = user_item_matrix.fillna(0)

    # Calculate similarity between users using cosine similarity
    if similarity_measure == 'cosine':
        similarity_matrix = cosine_similarity(user_item_matrix)
    else:
        raise ValueError(f"Invalid similarity measure: {similarity_measure}")

    similarity_matrix = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

    # Get the k most similar users to the given user
    similar_users = similarity_matrix[user_id].sort_values(ascending=False)[1:k + 1].index

    # Predict ratings for the user based on the similar users
    predictions = {}
    for item in user_item_matrix.columns:
        weighted_sum = 0
        sum_of_weights = 0
        for user in similar_users:
            if user_item_matrix[item][user] > 0:
                weight = similarity_matrix[user_id][user]
                weighted_sum += weight * user_item_matrix[item][user]
                sum_of_weights += weight
        if sum_of_weights > 0:
            predictions[item] = weighted_sum / sum_of_weights
        else:
            predictions[item] = 0

    return predictions


# Set the user for which we want to predict mineral sites
user_id = 'F'

# Apply collaborative filtering
predictions = collaborative_filtering(df, user_id, k=3)

# Print the predicted scores for each mineral
print("Predicted Mineral Scores for User 'F':")
for mineral, score in predictions.items():
    print(f"{mineral}: {score:.2f}")

# Threshold for classification
threshold = 0.5

# Classify predicted sites as potential mineral sites
print("\nPotential Mineral Sites:")
for mineral, score in predictions.items():
    if score >= threshold:
        print(f"{mineral}")
