from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import uvicorn

app = FastAPI(
    title="Embedding Clustering API",
    description="API for clustering embeddings using DBSSCAN based on cosine similarity",
)

class EmbeddingItem(BaseModel):
    """A single embedding vector with its ID"""
    vector: List[float] = Field(..., description="The embedding vector")


class EmbeddingsInput(BaseModel):
    """Input model with a dictionary of embeddings where keys are IDs and values are embedding vectors"""
    embeddings: Dict[str, List[float]] = Field(
        ...,
        description="Dictionary mapping IDs to embedding vectors"
    )
    similarity_threshold: Optional[float] = Field(
        0.55,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for clustering (between 0 and 1)"
    )


class ClusterOutput(BaseModel):
    """Output model with clusters and their corresponding IDs"""
    clusters: Dict[str, List[str]] = Field(
        ...,
        description="Dictionary mapping cluster labels to lists of IDs"
    )


def get_cluster(id_embedding_dict, similarity=0.55):
    """
    Cluster embeddings using DBSCAN based on cosine similarity.
    
    Parameters
    ----------
    id_embedding_dict : dict
        A dictionary mapping IDs to embedding vectors
    
    similarity : float, default=0.55
        The cosine similarity threshold for clustering
    
    Returns
    -------
    dict
        A dictionary mapping cluster labels to lists of IDs
    """
    # Extract IDs and embeddings
    ids = list(id_embedding_dict.keys())
    embeddings = [np.array(id_embedding_dict[id_]) for id_ in ids]
    
    # Convert embeddings to a 2D numpy array
    embeddings_array = np.vstack(embeddings)

    # Calculate the cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings_array)

    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1 - similarity_matrix

    # Ensure no negative values (which can happen due to floating point precision)
    distance_matrix = np.maximum(distance_matrix, 0)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=1-similarity, min_samples=1, metric='precomputed')
    cluster_labels = dbscan.fit_predict(distance_matrix)

    # Group IDs by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        label_str = str(label)  # Convert label to string for JSON compatibility
        if label_str not in clusters:
            clusters[label_str] = []
        clusters[label_str].append(ids[i])

    return clusters


@app.post("/cluster", response_model=ClusterOutput)
async def cluster_embeddings(input_data: EmbeddingsInput):
    """
    Cluster embeddings based on cosine similarity
    
    Takes a dictionary of embeddings with their IDs and returns clusters of IDs
    """
    try:
        clusters = get_cluster(
            input_data.embeddings, 
            similarity=input_data.similarity_threshold
        )
        return ClusterOutput(clusters=clusters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")


@app.get("/")
async def root():
    """API root - returns basic information about the API"""
    return {
        "message": "Embedding Clustering API",
        "docs": "/docs",
        "usage": "Send a POST request to /cluster with embedding data"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)