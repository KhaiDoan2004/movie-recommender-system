"""
Script tạo Hybrid Embeddings (Content + Collaborative)
Thay thế TF-IDF bằng embeddings nâng cao:
1. Content Embeddings: Sentence-BERT
2. Collaborative Embeddings: SVD (Matrix Factorization)
3. Hybrid Embeddings: Kết hợp cả 2
"""

import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from surprise import SVD, Dataset, Reader
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')

# Đường dẫn - relative từ project root
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = str(BASE_DIR / "data_cleaned")
OUTPUT_DIR = str(BASE_DIR / "embeddings")
MODELS_DIR = str(BASE_DIR / "models")

# Tạo thư mục nếu chưa có
for dir_path in [OUTPUT_DIR, MODELS_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_cleaned_data():
    """Load dữ liệu đã làm sạch"""
    print("=" * 80)
    print("LOADING CLEANED DATA...")
    print("=" * 80)
    
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies_cleaned.csv"))
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings_cleaned.csv"))
    
    print(f"✅ Loaded {len(movies)} movies")
    print(f"✅ Loaded {len(ratings)} ratings")
    
    return movies, ratings

def create_content_embeddings(movies):
    """
    Tạo Content Embeddings bằng Sentence-BERT
    Thay thế TF-IDF bằng embeddings nâng cao
    """
    print("\n" + "=" * 80)
    print("CREATING CONTENT EMBEDDINGS (Sentence-BERT)")
    print("=" * 80)
    
    print("\n1. Loading Sentence-BERT model...")
    # Model nhỏ, nhanh, đủ tốt cho recommendation
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("   ✅ Model loaded: all-MiniLM-L6-v2 (384 dimensions)")
    
    print("\n2. Creating embeddings from content_text...")
    content_texts = movies['content_text'].tolist()
    
    print(f"   - Processing {len(content_texts)} movies...")
    print("   - This may take a few minutes...")
    
    # Tạo embeddings (batch processing tự động)
    content_embeddings = model.encode(
        content_texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    
    print(f"\n✅ Content embeddings created: {content_embeddings.shape}")
    print(f"   - Shape: ({len(movies)}, 384)")
    print(f"   - Sample embedding (first 5 values): {content_embeddings[0][:5]}")
    
    # Lưu model và embeddings
    model_path = os.path.join(MODELS_DIR, "sentence_bert_model")
    model.save(model_path)
    print(f"   - Model saved to: {model_path}")
    
    np.save(os.path.join(OUTPUT_DIR, "content_embeddings.npy"), content_embeddings)
    print(f"   - Embeddings saved to: {OUTPUT_DIR}/content_embeddings.npy")
    
    return content_embeddings, model

def create_collaborative_embeddings(ratings, n_factors=50):
    """
    Tạo Collaborative Embeddings bằng SVD (Matrix Factorization)
    Học từ rating patterns của users
    """
    print("\n" + "=" * 80)
    print("CREATING COLLABORATIVE EMBEDDINGS (SVD)")
    print("=" * 80)
    
    print("\n1. Preparing data for SVD...")
    # Chuẩn bị data cho Surprise
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(
        ratings[['userId', 'movieId', 'rating']],
        reader
    )
    trainset = data.build_full_trainset()
    
    print(f"   - Users: {trainset.n_users}")
    print(f"   - Items (movies): {trainset.n_items}")
    print(f"   - Ratings: {trainset.n_ratings}")
    
    print(f"\n2. Training SVD model (n_factors={n_factors})...")
    print("   - This may take a few minutes...")
    
    # Train SVD
    algo = SVD(n_factors=n_factors, random_state=42, verbose=False)
    algo.fit(trainset)
    
    print("   ✅ SVD model trained!")
    
    # Lấy item embeddings (qi trong SVD)
    # qi là item factors (embeddings cho mỗi movie)
    print("\n3. Extracting item embeddings...")
    
    # Cần map movieId sang internal item id của Surprise
    # Tạo mapping
    movie_id_to_inner_id = {trainset.to_raw_iid(i): i for i in range(trainset.n_items)}
    
    # Lấy embeddings cho tất cả items
    item_embeddings = np.array([algo.qi[i] for i in range(trainset.n_items)])
    
    print(f"   ✅ Item embeddings extracted: {item_embeddings.shape}")
    print(f"   - Shape: ({trainset.n_items}, {n_factors})")
    print(f"   - Sample embedding (first 5 values): {item_embeddings[0][:5]}")
    
    # Lưu model và embeddings
    model_path = os.path.join(MODELS_DIR, "svd_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(algo, f)
    
    # Lưu mapping
    mapping_path = os.path.join(MODELS_DIR, "movie_id_mapping.pkl")
    with open(mapping_path, 'wb') as f:
        pickle.dump({
            'movie_id_to_inner_id': movie_id_to_inner_id,
            'inner_id_to_movie_id': {v: k for k, v in movie_id_to_inner_id.items()}
        }, f)
    
    print(f"   - Model saved to: {model_path}")
    print(f"   - Mapping saved to: {mapping_path}")
    
    np.save(os.path.join(OUTPUT_DIR, "collaborative_embeddings.npy"), item_embeddings)
    print(f"   - Embeddings saved to: {OUTPUT_DIR}/collaborative_embeddings.npy")
    
    return item_embeddings, algo, movie_id_to_inner_id

def create_hybrid_embeddings(content_emb, collab_emb, movies, movie_id_mapping):
    """
    Kết hợp Content và Collaborative Embeddings thành Hybrid Embeddings
    """
    print("\n" + "=" * 80)
    print("CREATING HYBRID EMBEDDINGS")
    print("=" * 80)
    
    print("\n1. Aligning embeddings...")
    # Content embeddings có cho tất cả movies
    # Collaborative embeddings chỉ có cho movies có ratings
    
    # Tạo matrix cho tất cả movies
    n_movies = len(movies)
    content_dim = content_emb.shape[1]  # 384
    collab_dim = collab_emb.shape[1]    # 50
    
    print(f"   - Total movies: {n_movies}")
    print(f"   - Content dim: {content_dim}")
    print(f"   - Collaborative dim: {collab_dim}")
    
    # Tạo full collaborative embeddings matrix
    # Movies không có rating → zero vector
    full_collab_emb = np.zeros((n_movies, collab_dim))
    
    for idx, row in movies.iterrows():
        movie_id = row['movieId']
        if movie_id in movie_id_mapping:
            inner_id = movie_id_mapping[movie_id]
            if inner_id < len(collab_emb):
                full_collab_emb[idx] = collab_emb[inner_id]
    
    print(f"   - Movies with collaborative embeddings: {(full_collab_emb.sum(axis=1) != 0).sum()}")
    
    print("\n2. Normalizing embeddings...")
    # Normalize để tránh scale khác nhau
    content_emb_norm = normalize(content_emb, norm='l2', axis=1)
    collab_emb_norm = normalize(full_collab_emb, norm='l2', axis=1)
    
    print("\n3. Creating hybrid embeddings (concatenate)...")
    # Option 1: Concatenate (giữ nguyên cả 2)
    hybrid_emb_concat = np.concatenate([content_emb_norm, collab_emb_norm], axis=1)
    
    print(f"   ✅ Hybrid embeddings (concatenate): {hybrid_emb_concat.shape}")
    print(f"   - Shape: ({n_movies}, {content_dim + collab_dim}) = ({n_movies}, 434)")
    
    print("\n4. Creating hybrid embeddings (weighted average)...")
    # Option 2: Weighted average (giữ cùng dimension)
    # Chỉ dùng khi cả 2 có cùng dimension, hoặc project về cùng dimension
    # Ở đây ta dùng concatenate vì dimensions khác nhau
    
    # Lưu cả 2 loại
    np.save(os.path.join(OUTPUT_DIR, "hybrid_embeddings_concat.npy"), hybrid_emb_concat)
    print(f"   - Saved: {OUTPUT_DIR}/hybrid_embeddings_concat.npy")
    
    # Tạo metadata
    metadata = {
        'content_dim': content_dim,
        'collab_dim': collab_dim,
        'hybrid_dim': content_dim + collab_dim,
        'n_movies': n_movies,
        'method': 'concatenate'
    }
    
    import json
    with open(os.path.join(OUTPUT_DIR, "embeddings_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   - Metadata saved: {OUTPUT_DIR}/embeddings_metadata.json")
    
    return hybrid_emb_concat, metadata

def create_movie_embedding_mapping(movies, hybrid_emb):
    """
    Tạo mapping movieId -> embedding index
    """
    print("\n" + "=" * 80)
    print("CREATING MOVIE EMBEDDING MAPPING")
    print("=" * 80)
    
    mapping = {}
    for idx, row in movies.iterrows():
        mapping[int(row['movieId'])] = idx
    
    mapping_path = os.path.join(OUTPUT_DIR, "movie_embedding_mapping.pkl")
    with open(mapping_path, 'wb') as f:
        pickle.dump(mapping, f)
    
    print(f"✅ Mapping created: {len(mapping)} movies")
    print(f"   - Saved to: {mapping_path}")
    
    return mapping

def main():
    """Hàm chính"""
    print("\n" + "=" * 80)
    print("CREATING HYBRID EMBEDDINGS (Content + Collaborative)")
    print("=" * 80)
    print("\n⚠️  LƯU Ý: Quá trình này sẽ:")
    print("   1. Download Sentence-BERT model (lần đầu ~90MB)")
    print("   2. Train SVD model (vài phút)")
    print("   3. Tạo embeddings cho tất cả movies")
    print("   → Tổng thời gian: ~5-10 phút")
    print("=" * 80)
    
    # Load data
    movies, ratings = load_cleaned_data()
    
    # 1. Content embeddings
    content_emb, content_model = create_content_embeddings(movies)
    
    # 2. Collaborative embeddings
    collab_emb, svd_model, movie_id_mapping = create_collaborative_embeddings(ratings)
    
    # 3. Hybrid embeddings
    hybrid_emb, metadata = create_hybrid_embeddings(
        content_emb, collab_emb, movies, movie_id_mapping
    )
    
    # 4. Mapping
    embedding_mapping = create_movie_embedding_mapping(movies, hybrid_emb)
    
    # Tóm tắt
    print("\n" + "=" * 80)
    print("TÓM TẮT")
    print("=" * 80)
    print("\n✅ ĐÃ TẠO:")
    print(f"   1. Content Embeddings: {content_emb.shape} (Sentence-BERT)")
    print(f"   2. Collaborative Embeddings: {collab_emb.shape} (SVD)")
    print(f"   3. Hybrid Embeddings: {hybrid_emb.shape} (Concatenate)")
    print(f"   4. Models và mappings đã lưu")
    
    print("\n📁 FILES CREATED:")
    print(f"   - {OUTPUT_DIR}/content_embeddings.npy")
    print(f"   - {OUTPUT_DIR}/collaborative_embeddings.npy")
    print(f"   - {OUTPUT_DIR}/hybrid_embeddings_concat.npy")
    print(f"   - {OUTPUT_DIR}/embeddings_metadata.json")
    print(f"   - {OUTPUT_DIR}/movie_embedding_mapping.pkl")
    print(f"   - {MODELS_DIR}/sentence_bert_model/")
    print(f"   - {MODELS_DIR}/svd_model.pkl")
    print(f"   - {MODELS_DIR}/movie_id_mapping.pkl")
    
    print("\n📋 CÁC BƯỚC TIẾP THEO:")
    print("   1. Sử dụng hybrid_embeddings trong content_based.py")
    print("   2. Thay thế TF-IDF bằng hybrid embeddings")
    print("   3. Test và đánh giá chất lượng")
    
    print("\n" + "=" * 80)
    print("HOÀN THÀNH!")
    print("=" * 80)

if __name__ == "__main__":
    main()

