```python
from sentence_transformers import SentenceTransformer
import torch
import faiss
import pandas as pd
import os
from functions import clean_description

# Global mappings
sentenceToImg = {}  # Maps sentences to image names

# Global variables
model = None
sentences = None
embeddings = None
faiss_index = None
data = None

def create_and_save_embeddings():
    global data
    # Load data if not already loaded
    if data is None:
        data = pd.read_excel('./data.xlsx')
    descriptions = dict(zip(data.iloc[:, 0], data.iloc[:, 1]))
    clean_description(descriptions)

    # Build list of sentences
    sentences = []
    for key in descriptions.keys():
        sentences.append(descriptions[key])

    # Save sentences to file
    torch.save(sentences, './sentences.pt')

    # Generate embeddings using transformer model
    model_temp = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings_tensor = model_temp.encode(sentences, convert_to_tensor=True)

    # Normalize embeddings to unit vectors for cosine similarity
    normalized_embeddings = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)

    # Save embeddings to file
    torch.save(normalized_embeddings, './embeddings.pt')

    # Convert to NumPy and create FAISS index for cosine similarity (dot-product)
    embeddings_np = normalized_embeddings.cpu().numpy().astype('float32')
    index = faiss.IndexFlatIP(embeddings_np.shape[1])  # Dot-product index
    index.add(embeddings_np)
    faiss.write_index(index, 'index.faiss')

    print("Saving sentences, embeddings, and FAISS index (cosine) completed!")

def initialize():
    global sentenceToImg, model, sentences, embeddings, faiss_index

    # Create sentence-to-image mapping if not already created
    if not sentenceToImg:
        data = pd.read_excel('./data.xlsx')
        descriptions = dict(zip(data.iloc[:, 0], data.iloc[:, 1]))
        clean_description(descriptions)
        # Map descriptions to image names
        for key, value in descriptions.items():
            sentenceToImg[value] = key
        print("sentenceToImg loaded!")

    # Load model, sentences, embeddings, and FAISS index if not already loaded
    if model is None:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        print("Model loaded!")
    if sentences is None:
        sentences = torch.load('./sentences.pt')
        print("Sentences loaded!")
    if embeddings is None:
        embeddings = torch.load('./embeddings.pt')
        print("Embeddings loaded!")
    if faiss_index is None:
        faiss_index = faiss.read_index('index.faiss')
        print("FAISS index loaded!")

def getAllImages(description, min_similarity=0.5):
    # Initialize model and resources
    initialize()

    # Convert input description to embedding and normalize
    query_embedding = model.encode([description], convert_to_tensor=True)
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
    query_np = query_embedding.cpu().numpy().astype('float32')

    # Search FAISS index for all embeddings
    total_embeddings = faiss_index.ntotal
    distances, indices = faiss_index.search(query_np, total_embeddings)  # Distances and indices of nearest vectors

    # Filter results by similarity threshold
    filtered_sentences = []
    similarity_scores = []
    for score, idx in zip(distances[0], indices[0]):
        if score < min_similarity:
            break
        if idx < len(sentences):
            # Avoid duplicates
            if sentences[idx] not in filtered_sentences:
                filtered_sentences.append(sentences[idx])
                similarity_scores.append(round(float(score) * 100, 2))  # Convert to percentage
    return filtered_sentences, similarity_scores

def getImgByPath(description):
    # Get similar descriptions and their scores
    allimg, scores = getAllImages(description)

    # Verify and return existing images
    allImages = []
    for sent in allimg:
        img_name = sentenceToImg.get(sent, None)
        print(img_name)
        if img_name is not None:
            img_path = f"./static/{img_name}.jpg"
            if os.path.exists(img_path):
                print(f"The image {img_name} exists!")
                allImages.append(img_name)
            else:
                print(f"The image {img_name} does not exist!")
        else:
            print(f"No matching image found for the sentence: {sent}, image: {img_name}")
    return [allImages, allimg, scores]

def add_image_description(img_name, description):
    global sentences, embeddings, data, model, faiss_index

    # Load sentences if not already loaded
    if not sentences:
        if os.path.exists('./sentences.pt'):
            sentences = torch.load('./sentences.pt')
        else:
            sentences = []

    # Load embeddings if not already loaded
    if embeddings is None:
        if os.path.exists('./embeddings.pt'):
            embeddings = torch.load('./embeddings.pt')
        else:
            embeddings = torch.tensor([])

    # Add new description to sentences
    sentences.append(description)
    sentenceToImg[description] = img_name  # Update sentence-to-image mapping

    # Generate embedding for new description
    if model is None:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    new_embedding = model.encode(description, convert_to_tensor=True).unsqueeze(0)

    # Concatenate new embedding with existing ones
    if embeddings.numel() == 0:
        updated_embeddings = new_embedding
    else:
        updated_embeddings = torch.cat((embeddings, new_embedding), dim=0)

    # Normalize embeddings to unit vectors
    normalized_embeddings = torch.nn.functional.normalize(updated_embeddings, p=2, dim=1)
    embeddings = normalized_embeddings

    # Save updated sentences and embeddings
    torch.save(sentences, './sentences.pt')
    torch.save(normalized_embeddings, './embeddings.pt')

    # Update FAISS index
    embeddings_np = normalized_embeddings.cpu().numpy().astype('float32')
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    faiss.write_index(index, 'index.faiss')
    faiss_index = index

    # Load DataFrame if not already loaded
    if data is None or data.empty:
        if os.path.exists('./data.xlsx'):
            data = pd.read_excel('./data.xlsx')
        else:
            data = pd.DataFrame(columns=["A", "B"])

    # Add new image and description to DataFrame
    new_entry = pd.DataFrame({"A": [img_name], "B": [description]})
    updated_data = pd.concat([data, new_entry], ignore_index=True)

    # Save updated DataFrame
    updated_data.to_excel('./data.xlsx', index=False)
    data = updated_data

def remove_image_description(img_name):
    global sentences, embeddings, data, model, faiss_index

    # Load sentences and embeddings if not already loaded
    if not sentences:
        sentences = torch.load('./sentences.pt')
    if not embeddings:
        embeddings = torch.load('./embeddings.pt')

    # Remove description from sentences
    if img_name in data["A"].values:
        idx_to_remove = data[data["A"] == img_name].index
        description_to_remove = data.loc[idx_to_remove, "B"].tolist()
        sentences = [s for s in sentences if s not in description_to_remove]

        # Remove embedding for the description
        model_temp = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        model = model_temp
        indices_to_keep = [i for i, s in enumerate(sentences) if s not in description_to_remove]
        embeddings = embeddings[indices_to_keep]

        # Normalize embeddings to unit vectors
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        embeddings = normalized_embeddings

        # Save updated sentences and embeddings
        torch.save(sentences, './sentences.pt')
        torch.save(normalized_embeddings, './embeddings.pt')

        # Update FAISS index
        embeddings_np = normalized_embeddings.cpu().numpy().astype('float32')
        index = faiss.IndexFlatIP(embeddings_np.shape[1])
        index.add(embeddings_np)
        faiss.write_index(index, 'index.faiss')
        faiss_index = index

        if not data.empty:
            # Remove image and description from DataFrame
            updated_data = data.drop(idx_to_remove)
            updated_data.to_excel('./data.xlsx', index=False)
            data = updated_data
```