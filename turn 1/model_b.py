import pandas as pd

# Sample data
data = {
    'Region': ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8'],
    'Deposit': ['Gold', 'Gold', 'Copper', 'Silver', 'Gold', 'Copper', None, None],
    'Sulfur_Content': [10, 20, 15, 5, 18, 25, 8, 12],
    'Temperature': [25, 28, 30, 27, 26, 29, 24, 22],
    'Pressure': [1000, 800, 1200, 900, 1100, 950, 700, 850]
}

df = pd.DataFrame(data)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df.drop(columns=['Region', 'Deposit']))
df_scaled = pd.DataFrame(df_scaled, columns=df.drop(columns=['Region', 'Deposit']).columns)

from scipy.sparse import csr_matrix

sparse_matrix = csr_matrix(df_scaled.values)

from sklearn.neighbors import NearestNeighbors

knn = NearestNeighbors(metric='cosine', n_neighbors=3)
knn.fit(sparse_matrix)

# Example new region data (Needs to be scaled)
new_region_data = [20, 30, 8]
new_region_data = scaler.transform([new_region_data])

# Find KNN for the new region
distances, indices = knn.kneighbors(new_region_data, return_distance=True)

# Get the indices of the similar regions in the original dataset
similar_regions = indices[0]

# Predict the mineral deposits based on the KNN similarities
predicted_deposits = [df.iloc[i]['Deposit'] for i in similar_regions]

# Most frequent prediction
predicted_deposit = max(set(predicted_deposits), key=predicted_deposits.count)
