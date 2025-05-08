# Dhivehi Article Clustering API

A FastAPI-based API for clustering Dhivehi text embeddings using DBSCAN algorithm with cosine similarity.

## Installation

1. Clone this repository and create an env using Python 3.11
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the API:

```bash
uvicorn main:app --reload
```

The API will be available at http://localhost:8000

## Usage Example

### Python Client

```python
import requests

# Example embeddings
data = {
    "embeddings": {
        "doc1": [0.1, 0.2, 0.3],
        "doc2": [0.15, 0.25, 0.35],
        "doc3": [0.9, 0.8, 0.7]
    },
    "similarity_threshold": 0.55 # (optional, default: 0.55)
}

response = requests.post("http://localhost:8000/cluster", json=data)
results = response.json()
print(results)
# Output might look like: {"clusters": {"0": ["doc1", "doc2"], "1": ["doc3"]}}
```
- `clusters`: Dictionary mapping cluster labels to lists of document IDs

## Tuning the Similarity Threshold

The `similarity_threshold` parameter controls how similar embeddings must be to be grouped together:

- **Default (0.55)**: A balanced threshold that works well for most use cases
- **Higher values (e.g., 0.8)**: More strict clustering, requiring embeddings to be very similar
- **Lower values (e.g., 0.3)**: More lenient clustering, grouping embeddings with less similarity