# Snap AI-Powered Image Captioning and Semantic Search Engine

An intelligent image search system that combines **custom AI image captioning** with **semantic vector search**. The system automatically generates detailed descriptions for images using InceptionV3 with attention mechanism, then enables natural language search using SentenceTransformer and FAISS vector similarity.

## 🎯 Key Features

- **Custom AI Image Captioning**: Generate automatic descriptions using InceptionV3 + Attention mechanism
- **Semantic Search**: Search images using natural language (e.g., "sunset at the beach")
- **Vector Similarity**: Fast, scalable search using FAISS indexing
- **Real-time Processing**: Optimized for social media integration

## 🔍 How It Works

1. **Image Analysis**: Custom trained model analyzes uploaded images and generates detailed captions
2. **Vector Encoding**: Text descriptions are converted to high-dimensional vectors using SentenceTransformer
3. **FAISS Indexing**: Vectors are stored in optimized FAISS index for fast similarity search
4. **Semantic Matching**: User queries are encoded and matched against image descriptions using cosine similarity

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow/Keras, InceptionV3, Attention Mechanism
- **NLP**: SentenceTransformer (BERT-based), GloVe embeddings
- **Search**: FAISS vector similarity search
- **Data Processing**: Pandas, NumPy, OpenCV
- **Backend Ready**: Designed for MongoDB integration

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data (Required)
Download and place these files in the project root:
- **Flickr30k Dataset**: Create `flickr30k_images/` folder with your images
- **GloVe Embeddings**: Download `glove.6B.200d.txt` from [GloVe project](https://nlp.stanford.edu/projects/glove/)
- **Image Descriptions**: Create `data.xlsx` with image names (column A) and descriptions (column B)

### 3. Generate Image Captions
```bash
# Train the captioning model (optional - if starting from scratch)
python full_model.py

# Generate captions for new images
python result.py
```

### 4. Setup Search Engine
```bash
# Create embeddings and FAISS index (run once)
python SBERT_FAISS.py
```

### 5. Search Images
```python
from SBERT_FAISS import getImgByPath

# Search for images matching a description
results = getImgByPath("beautiful sunset at the beach")
print(results)
```

## 📁 Project Structure

```
├── full_model.py          # AI model training (InceptionV3 + Attention)
├── result.py              # Caption generation and display
├── SBERT_FAISS.py         # Semantic search engine
├── functions.py           # Data processing utilities
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── flickr30k_images/*     # Image dataset (not included)
├── data.xlsx*            # Image descriptions (not included)
├── sentences.pt*          # Precomputed search data (generated)
├── embeddings.pt*         # Precomputed embeddings (generated)
├── index.faiss*          # FAISS index (generated)
├── wordtoix.pkl*         # Vocabulary mapping (generated)
├── ixtoword.pkl*         # Reverse vocabulary (generated)
└── max_length.txt*       # Model parameter (generated)
```

**Files marked with asterisk (*) are not included in the repository**

## 🔧 API Usage

### Generate Caption for Image
```python
from result import getDescription

caption = getDescription("path/to/your/image.jpg")
print(f"Generated caption: {caption}")
```

### Search Images by Description
```python
from SBERT_FAISS import getImgByPath

# Find images matching semantic description
matching_images = getImgByPath("dog playing in park")
```

## 📊 Performance

- **Caption Generation**: ~2-3 seconds per image on CPU
- **Search Speed**: Sub-second search across thousands of images
- **Accuracy**: Semantic search finds relevant images even without exact keyword matches

## 🎨 Example Use Cases

- **Social Media Platforms**: Auto-tag uploaded images and enable smart search
- **E-commerce**: Search product images by description
- **Digital Asset Management**: Organize and find images in large databases
- **Content Discovery**: Recommend similar images based on semantic content

## 📋 Data Requirements

The following files are **not included** and must be provided:

| File | Description | Source |
|------|-------------|---------|
| `flickr30k_images/` | Image dataset | [Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/) or your own |
| `data.xlsx` | Image names + descriptions | Create manually or use Flickr30k annotations |
| `glove.6B.200d.txt` | Word embeddings | [GloVe Project](https://nlp.stanford.edu/projects/glove/) |

**Generated files** (created automatically when you run the scripts):
- `sentences.pt, embeddings.pt, index.faiss` - Precomputed search data (generated by running SBERT_FAISS.py)
- `wordtoix.pkl, ixtoword.pkl, max_length.txt` - Precomputed captioning data (generated by running full_model.py)

**Note**: These files are generated based on your own dataset and are not included in the repository.

## 🚧 Future Enhancements

- [ ] Real-time image processing API
- [ ] Multi-language caption support
- [ ] Advanced filtering options
- [ ] Integration with cloud storage
- [ ] Web interface for demo

## 📧 Contact

For questions or collaboration opportunities: **mireuven123@gmail.com**

## 📄 License

MIT License - see LICENSE file for details.

---

*This project demonstrates advanced AI/ML capabilities including custom model training, vector search, and semantic understanding - perfect for modern computer vision applications.*