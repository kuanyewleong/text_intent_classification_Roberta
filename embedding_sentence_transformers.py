import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the dataset from a CSV file
data = pd.read_csv('input_dataset.csv')

# Load the sentence-transformers model
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Embed the text data using the model
embeddings = model.encode(data['text'].tolist())

# Create a new DataFrame with the embedded results
embedded_data = pd.DataFrame(embeddings, columns=['embedding_{}'.format(i) for i in range(embeddings.shape[1])])

# Concatenate the embedded data with the original data
output_data = pd.concat([data, embedded_data], axis=1)

# Save the output data to a new CSV file
output_data.to_csv('output_dataset.csv', index=False)
