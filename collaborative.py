"""
Collaborative Filtering Recommender
Sử dụng Collaborative Embeddings (SVD) để recommend dựa trên user ratings
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from surprise import SVD, Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity

# Đường dẫn - relative từ project root
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = str(BASE_DIR / "data_cleaned")
EMBEDDINGS_DIR = str(BASE_DIR / "embeddings")
MODELS_DIR = str(BASE_DIR / "models")

class CollaborativeRecommender:
    """Collaborative Filtering Recommender sử dụng SVD"""
    
    def __init__(self):
        """Khởi tạo recommender"""
        print("=" * 80)
        print("INITIALIZING COLLABORATIVE RECOMMENDER")
        print("=" * 80)
        
        # Load movies và ratings
        self.movies = pd.read_csv(os.path.join(DATA_DIR, "movies_cleaned.csv"))
        self.ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings_cleaned.csv"))
        print(f"✅ Loaded {len(self.movies)} movies")
        print(f"✅ Loaded {len(self.ratings)} ratings")
        
        # Load SVD model
        print("\n📦 Loading SVD model...")
        with open(os.path.join(MODELS_DIR, "svd_model.pkl"), 'rb') as f:
            self.svd_model = pickle.load(f)
        print("   ✅ SVD model loaded")
        
        # Load collaborative embeddings
        print("\n📦 Loading Collaborative Embeddings...")
        self.collab_embeddings = np.load(os.path.join(EMBEDDINGS_DIR, "collaborative_embeddings.npy"))
        print(f"   ✅ Collaborative embeddings loaded: {self.collab_embeddings.shape}")
        
        # Load mappings
        with open(os.path.join(MODELS_DIR, "movie_id_mapping.pkl"), 'rb') as f:
            self.mappings = pickle.load(f)
        
        self.movie_id_to_inner_id = self.mappings['movie_id_to_inner_id']
        self.inner_id_to_movie_id = self.mappings['inner_id_to_movie_id']
        print(f"✅ Loaded movie ID mappings")
        
        # Tạo rating matrix (sparse) để tính item-item similarity
        self.rating_matrix = None
        self._build_rating_matrix()
    
    def _build_rating_matrix(self):
        """Xây dựng rating matrix (user x movie)"""
        print("\n🔄 Building rating matrix...")
        # Tạo pivot table
        self.rating_matrix = self.ratings.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )
        print(f"   ✅ Rating matrix: {self.rating_matrix.shape}")
    
    def predict_rating(self, user_id, movie_id):
        """
        Dự đoán rating của user cho movie (dùng SVD model)
        
        Args:
            user_id: ID của user
            movie_id: ID của phim
        
        Returns:
            Predicted rating (0.5-5.0)
        """
        try:
            prediction = self.svd_model.predict(user_id, movie_id)
            # Clip về range 0.5-5.0
            rating = max(0.5, min(5.0, prediction.est))
            return rating
        except:
            return None
    
    def recommend_for_user(self, user_rated_movies, n=10, min_rating=3.5):
        """
        Recommend phim cho user dựa trên ratings đã có
        
        Args:
            user_rated_movies: Dict {movie_id: rating}
            n: Số lượng recommendations
            min_rating: Rating tối thiểu để recommend
        
        Returns:
            DataFrame với recommendations
        """
        if len(user_rated_movies) == 0:
            return pd.DataFrame()
        
        # Tạo user profile từ ratings
        # Tính average embedding của các phim đã rate (weighted by rating)
        embeddings_list = []
        weights = []
        
        for movie_id, rating in user_rated_movies.items():
            if movie_id in self.movie_id_to_inner_id:
                inner_id = self.movie_id_to_inner_id[movie_id]
                if inner_id < len(self.collab_embeddings):
                    embeddings_list.append(self.collab_embeddings[inner_id])
                    weights.append(rating)  # Weight theo rating
        
        if len(embeddings_list) == 0:
            return pd.DataFrame()
        
        # Weighted average embedding
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        user_profile = np.average(embeddings_list, axis=0, weights=weights).reshape(1, -1)
        
        # Tính similarity với tất cả phim
        similarities = cosine_similarity(user_profile, self.collab_embeddings)[0]
        
        # Loại bỏ các phim đã rate
        for movie_id in user_rated_movies.keys():
            if movie_id in self.movie_id_to_inner_id:
                inner_id = self.movie_id_to_inner_id[movie_id]
                if inner_id < len(similarities):
                    similarities[inner_id] = -1
        
        # Lấy top n
        top_indices = np.argsort(similarities)[::-1][:n]
        
        # Tạo kết quả
        results = []
        for inner_id in top_indices:
            if similarities[inner_id] > 0:
                if inner_id in self.inner_id_to_movie_id:
                    movie_id = int(self.inner_id_to_movie_id[inner_id])
                    movie = self.movies[self.movies['movieId'] == movie_id]
                    if len(movie) > 0:
                        movie = movie.iloc[0].copy()
                        movie['similarity_score'] = similarities[inner_id]
                        # Dự đoán rating (nếu có user_id)
                        # movie['predicted_rating'] = self.predict_rating(user_id, movie_id)
                        results.append(movie)
        
        if len(results) == 0:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('similarity_score', ascending=False)
        
        return results_df
    
    def get_item_based_recommendations(self, movie_id, n=10):
        """
        Item-based collaborative filtering
        Tìm phim tương tự dựa trên collaborative embeddings
        
        Args:
            movie_id: ID của phim
            n: Số lượng recommendations
        
        Returns:
            DataFrame với recommendations
        """
        if movie_id not in self.movie_id_to_inner_id:
            return pd.DataFrame()
        
        inner_id = self.movie_id_to_inner_id[movie_id]
        if inner_id >= len(self.collab_embeddings):
            return pd.DataFrame()
        
        # Tính similarity
        movie_emb = self.collab_embeddings[inner_id].reshape(1, -1)
        similarities = cosine_similarity(movie_emb, self.collab_embeddings)[0]
        
        # Loại bỏ chính nó
        similarities[inner_id] = -1
        
        # Lấy top n
        top_indices = np.argsort(similarities)[::-1][:n]
        
        # Tạo kết quả
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                if idx in self.inner_id_to_movie_id:
                    similar_movie_id = int(self.inner_id_to_movie_id[idx])
                    movie = self.movies[self.movies['movieId'] == similar_movie_id]
                    if len(movie) > 0:
                        movie = movie.iloc[0].copy()
                        movie['similarity_score'] = similarities[idx]
                        results.append(movie)
        
        if len(results) == 0:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('similarity_score', ascending=False)
        
        return results_df
    
    def get_user_based_recommendations(self, user_id, n=10):
        """
        User-based collaborative filtering
        Tìm users tương tự và recommend phim họ đã rate cao
        
        Args:
            user_id: ID của user
            n: Số lượng recommendations
        
        Returns:
            DataFrame với recommendations
        """
        if user_id not in self.rating_matrix.index:
            return pd.DataFrame()
        
        # Lấy ratings của user
        user_ratings = self.rating_matrix.loc[user_id]
        
        # Tìm users tương tự (cosine similarity trên rating vectors)
        user_vector = user_ratings.values.reshape(1, -1)
        similarities = cosine_similarity(user_vector, self.rating_matrix.values)[0]
        
        # Loại bỏ chính user đó
        user_idx = self.rating_matrix.index.get_loc(user_id)
        similarities[user_idx] = -1
        
        # Lấy top similar users
        top_user_indices = np.argsort(similarities)[::-1][:5]  # Top 5 similar users
        
        # Tổng hợp phim từ similar users (weighted by similarity)
        movie_scores = {}
        for user_idx in top_user_indices:
            if similarities[user_idx] > 0:
                similar_user_id = self.rating_matrix.index[user_idx]
                similar_user_ratings = self.rating_matrix.loc[similar_user_id]
                
                # Chỉ lấy phim user chưa rate
                for movie_id, rating in similar_user_ratings.items():
                    if rating > 0 and user_ratings[movie_id] == 0:  # User chưa rate
                        if movie_id not in movie_scores:
                            movie_scores[movie_id] = 0
                        movie_scores[movie_id] += rating * similarities[user_idx]
        
        # Sắp xếp và lấy top n
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # Tạo kết quả
        results = []
        for movie_id, score in sorted_movies:
            movie = self.movies[self.movies['movieId'] == movie_id]
            if len(movie) > 0:
                movie = movie.iloc[0].copy()
                movie['recommendation_score'] = score
                results.append(movie)
        
        if len(results) == 0:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        return results_df

def main():
    """Test collaborative recommender"""
    print("\n" + "=" * 80)
    print("TESTING COLLABORATIVE RECOMMENDER")
    print("=" * 80)
    
    # Khởi tạo recommender
    recommender = CollaborativeRecommender()
    
    # Test 1: Item-based recommendations
    print("\n1. Item-based recommendations for 'Toy Story' (movieId=1)...")
    recommendations = recommender.get_item_based_recommendations(movie_id=1, n=10)
    if len(recommendations) > 0:
        print(f"   Found {len(recommendations)} recommendations:")
        for idx, row in recommendations.head(5).iterrows():
            print(f"   - {row['title_clean']} ({row['year']:.0f}) - Similarity: {row['similarity_score']:.3f}")
    
    # Test 2: User-based recommendations
    print("\n2. User-based recommendations for user 1...")
    recommendations = recommender.get_user_based_recommendations(user_id=1, n=10)
    if len(recommendations) > 0:
        print(f"   Found {len(recommendations)} recommendations:")
        for idx, row in recommendations.head(5).iterrows():
            print(f"   - {row['title_clean']} ({row['year']:.0f}) - Score: {row['recommendation_score']:.3f}")
    
    # Test 3: Recommend for new user (có ratings)
    print("\n3. Recommendations for new user (rated movies: {1: 5.0, 2: 4.0, 3: 4.5})...")
    user_history = {1: 5.0, 2: 4.0, 3: 4.5}
    recommendations = recommender.recommend_for_user(user_history, n=10)
    if len(recommendations) > 0:
        print(f"   Found {len(recommendations)} recommendations:")
        for idx, row in recommendations.head(5).iterrows():
            print(f"   - {row['title_clean']} ({row['year']:.0f}) - Similarity: {row['similarity_score']:.3f}")
    
    print("\n" + "=" * 80)
    print("✅ Collaborative Recommender Test Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

