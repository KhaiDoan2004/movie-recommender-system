"""
Content-Based Recommender
Sử dụng Hybrid Embeddings để tìm phim tương tự dựa trên nội dung
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Đường dẫn - relative từ project root
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = str(BASE_DIR / "data_cleaned")
EMBEDDINGS_DIR = str(BASE_DIR / "embeddings")
MODELS_DIR = str(BASE_DIR / "models")

class ContentBasedRecommender:
    """Content-based recommender sử dụng Hybrid Embeddings"""
    
    def __init__(self, use_hybrid=True):
        """
        Khởi tạo recommender
        
        Args:
            use_hybrid: True = dùng hybrid embeddings, False = chỉ dùng content embeddings
        """
        print("=" * 80)
        print("INITIALIZING CONTENT-BASED RECOMMENDER")
        print("=" * 80)
        
        # Load movies data
        self.movies = pd.read_csv(os.path.join(DATA_DIR, "movies_cleaned.csv"))
        print(f"✅ Loaded {len(self.movies)} movies")
        
        # Load embeddings
        if use_hybrid:
            print("\n📦 Loading Hybrid Embeddings...")
            self.embeddings = np.load(os.path.join(EMBEDDINGS_DIR, "hybrid_embeddings_concat.npy"))
            print(f"   ✅ Hybrid embeddings loaded: {self.embeddings.shape}")
        else:
            print("\n📦 Loading Content Embeddings...")
            self.embeddings = np.load(os.path.join(EMBEDDINGS_DIR, "content_embeddings.npy"))
            print(f"   ✅ Content embeddings loaded: {self.embeddings.shape}")
        
        # Load movie embedding mapping
        with open(os.path.join(EMBEDDINGS_DIR, "movie_embedding_mapping.pkl"), 'rb') as f:
            self.movie_to_idx = pickle.load(f)
        
        print(f"✅ Loaded movie embedding mapping: {len(self.movie_to_idx)} movies")
        
        # Pre-compute similarity matrix (optional, để tăng tốc)
        self.similarity_matrix = None
        self.use_precomputed = False
    
    def precompute_similarity_matrix(self):
        """Pre-compute similarity matrix để tăng tốc (tốn RAM nhưng nhanh hơn)"""
        print("\n🔄 Pre-computing similarity matrix...")
        print("   This may take a few minutes...")
        self.similarity_matrix = cosine_similarity(self.embeddings)
        self.use_precomputed = True
        print("   ✅ Similarity matrix computed!")
    
    def get_similar_movies(self, movie_id, n=10, exclude_self=True):
        """
        Tìm n phim tương tự với movie_id
        
        Args:
            movie_id: ID của phim
            n: Số lượng phim tương tự cần tìm
            exclude_self: Có loại bỏ chính phim đó không
        
        Returns:
            DataFrame với các phim tương tự
        """
        # Kiểm tra movie_id có trong mapping không
        if movie_id not in self.movie_to_idx:
            print(f"⚠️  Movie ID {movie_id} not found in embeddings")
            return pd.DataFrame()
        
        # Lấy index của phim
        movie_idx = self.movie_to_idx[movie_id]
        
        # Tính similarity
        if self.use_precomputed and self.similarity_matrix is not None:
            # Dùng pre-computed matrix (nhanh)
            similarities = self.similarity_matrix[movie_idx]
        else:
            # Tính similarity on-the-fly
            movie_emb = self.embeddings[movie_idx].reshape(1, -1)
            similarities = cosine_similarity(movie_emb, self.embeddings)[0]
        
        # Lấy top n similar (bỏ chính nó nếu exclude_self)
        if exclude_self:
            similarities[movie_idx] = -1  # Set similarity với chính nó = -1
        
        top_indices = np.argsort(similarities)[::-1][:n]
        
        # Tạo kết quả
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Chỉ lấy similarity > 0
                movie_idx_in_df = idx  # Index trong embeddings = index trong movies (nếu align đúng)
                if movie_idx_in_df < len(self.movies):
                    movie = self.movies.iloc[movie_idx_in_df].copy()
                    movie['similarity_score'] = similarities[idx]
                    results.append(movie)
        
        if len(results) == 0:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('similarity_score', ascending=False)
        
        return results_df
    
    def get_similar_movies_by_title(self, title, n=10):
        """
        Tìm phim tương tự dựa trên title (tìm movie_id trước)
        
        Args:
            title: Tên phim
            n: Số lượng phim tương tự
        
        Returns:
            DataFrame với các phim tương tự
        """
        # Tìm movie_id từ title
        movie = self.movies[self.movies['title_clean'].str.lower() == title.lower()]
        if len(movie) == 0:
            # Thử fuzzy match
            from fuzzywuzzy import process
            titles = self.movies['title_clean'].tolist()
            match = process.extractOne(title, titles, scorer= fuzz.partial_ratio)
            if match and match[1] >= 80:
                movie = self.movies[self.movies['title_clean'] == match[0]]
        
        if len(movie) == 0:
            print(f"⚠️  Movie '{title}' not found")
            return pd.DataFrame()
        
        movie_id = movie.iloc[0]['movieId']
        return self.get_similar_movies(movie_id, n=n)
    
    def recommend_for_user_content_only(self, user_rated_movies, n=10):
        """
        Recommend cho user dựa trên các phim đã rate (chỉ dùng content)
        
        Args:
            user_rated_movies: Dict {movie_id: rating} hoặc List of movie_ids
            n: Số lượng recommendations
        
        Returns:
            DataFrame với recommendations
        """
        if isinstance(user_rated_movies, dict):
            movie_ids = list(user_rated_movies.keys())
        else:
            movie_ids = user_rated_movies
        
        if len(movie_ids) == 0:
            return pd.DataFrame()
        
        # Tính average embedding của các phim đã rate
        embeddings_list = []
        for movie_id in movie_ids:
            if movie_id in self.movie_to_idx:
                idx = self.movie_to_idx[movie_id]
                embeddings_list.append(self.embeddings[idx])
        
        if len(embeddings_list) == 0:
            return pd.DataFrame()
        
        # Average embedding
        user_profile = np.mean(embeddings_list, axis=0).reshape(1, -1)
        
        # Tính similarity với tất cả phim
        similarities = cosine_similarity(user_profile, self.embeddings)[0]
        
        # Loại bỏ các phim đã rate
        for movie_id in movie_ids:
            if movie_id in self.movie_to_idx:
                idx = self.movie_to_idx[movie_id]
                similarities[idx] = -1
        
        # Lấy top n
        top_indices = np.argsort(similarities)[::-1][:n]
        
        # Tạo kết quả
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                if idx < len(self.movies):
                    movie = self.movies.iloc[idx].copy()
                    movie['similarity_score'] = similarities[idx]
                    results.append(movie)
        
        if len(results) == 0:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('similarity_score', ascending=False)
        
        return results_df

def main():
    """Test content-based recommender"""
    print("\n" + "=" * 80)
    print("TESTING CONTENT-BASED RECOMMENDER")
    print("=" * 80)
    
    # Khởi tạo recommender
    recommender = ContentBasedRecommender(use_hybrid=True)
    
    # Test 1: Tìm similar movies cho Toy Story
    print("\n1. Finding similar movies to 'Toy Story' (movieId=1)...")
    similar = recommender.get_similar_movies(movie_id=1, n=10)
    if len(similar) > 0:
        print(f"   Found {len(similar)} similar movies:")
        for idx, row in similar.head(5).iterrows():
            print(f"   - {row['title_clean']} ({row['year']:.0f}) - Similarity: {row['similarity_score']:.3f}")
    
    # Test 2: Tìm bằng title
    print("\n2. Finding similar movies by title 'Avatar'...")
    similar = recommender.get_similar_movies_by_title("Avatar", n=5)
    if len(similar) > 0:
        print(f"   Found {len(similar)} similar movies:")
        for idx, row in similar.iterrows():
            print(f"   - {row['title_clean']} ({row['year']:.0f}) - Similarity: {row['similarity_score']:.3f}")
    
    # Test 3: Recommend cho user (content-only)
    print("\n3. Content-based recommendations for user (rated movies: [1, 2, 3])...")
    user_history = {1: 5.0, 2: 4.0, 3: 4.5}  # Toy Story, Jumanji, Grumpier Old Men
    recommendations = recommender.recommend_for_user_content_only(user_history, n=10)
    if len(recommendations) > 0:
        print(f"   Found {len(recommendations)} recommendations:")
        for idx, row in recommendations.head(5).iterrows():
            print(f"   - {row['title_clean']} ({row['year']:.0f}) - Similarity: {row['similarity_score']:.3f}")
    
    print("\n" + "=" * 80)
    print("✅ Content-Based Recommender Test Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

