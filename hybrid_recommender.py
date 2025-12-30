"""
Hybrid Recommender
Kết hợp Content-Based và Collaborative Filtering
Trọng số động dựa trên số lượng ratings của user
"""

import pandas as pd
import numpy as np
from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender

class HybridRecommender:
    """Hybrid Recommender kết hợp Content + Collaborative"""
    
    def __init__(self):
        """Khởi tạo hybrid recommender"""
        print("=" * 80)
        print("INITIALIZING HYBRID RECOMMENDER")
        print("=" * 80)
        
        # Khởi tạo các recommenders
        print("\n📦 Initializing Content-Based Recommender...")
        self.content_recommender = ContentBasedRecommender(use_hybrid=True)
        
        print("\n📦 Initializing Collaborative Recommender...")
        self.collab_recommender = CollaborativeRecommender()
        
        print("\n✅ Hybrid Recommender initialized!")
    
    def _calculate_alpha(self, num_ratings):
        """
        Tính trọng số alpha dựa trên số lượng ratings
        
        Args:
            num_ratings: Số lượng ratings của user
        
        Returns:
            alpha: Trọng số cho content-based (1-alpha cho collaborative)
        """
        if num_ratings == 0:
            return 1.0  # Chỉ dùng content
        elif num_ratings < 3:
            return 0.7  # Ưu tiên content
        elif num_ratings < 10:
            return 0.5  # Cân bằng
        elif num_ratings < 20:
            return 0.3  # Ưu tiên collaborative
        else:
            return 0.2  # Rất ưu tiên collaborative
    
    def recommend(self, user_rated_movies, n=10, use_content_weight=True):
        """
        Hybrid recommendation cho user
        
        Args:
            user_rated_movies: Dict {movie_id: rating}
            n: Số lượng recommendations
            use_content_weight: Có điều chỉnh trọng số theo số ratings không
        
        Returns:
            DataFrame với recommendations
        """
        num_ratings = len(user_rated_movies)
        
        # Tính alpha (trọng số)
        if use_content_weight:
            alpha = self._calculate_alpha(num_ratings)
        else:
            alpha = 0.5  # Cân bằng
        
        print(f"\n🔀 Hybrid Recommendation (alpha={alpha:.2f}, {num_ratings} ratings)")
        
        # Cold start: chỉ dùng content
        if num_ratings == 0:
            print("   → Cold start: Using popular movies")
            from search_engine import MovieSearchEngine
            search_engine = MovieSearchEngine()
            return search_engine.get_popular_movies(limit=n)
        
        # Lấy recommendations từ cả 2 methods
        print("   → Getting content-based recommendations...")
        content_recs = self.content_recommender.recommend_for_user_content_only(
            user_rated_movies, n=n*2  # Lấy nhiều hơn để merge
        )
        
        print("   → Getting collaborative recommendations...")
        collab_recs = self.collab_recommender.recommend_for_user(
            user_rated_movies, n=n*2  # Lấy nhiều hơn để merge
        )
        
        # Merge và tính điểm hybrid
        movie_scores = {}
        
        # Content-based scores
        if len(content_recs) > 0:
            max_content_score = content_recs['similarity_score'].max()
            for idx, row in content_recs.iterrows():
                movie_id = row['movieId']
                score = row['similarity_score'] / max_content_score if max_content_score > 0 else 0
                if movie_id not in movie_scores:
                    movie_scores[movie_id] = {'content': 0, 'collab': 0, 'movie': row}
                movie_scores[movie_id]['content'] = score
        
        # Collaborative scores
        if len(collab_recs) > 0:
            max_collab_score = collab_recs['similarity_score'].max()
            for idx, row in collab_recs.iterrows():
                movie_id = row['movieId']
                score = row['similarity_score'] / max_collab_score if max_collab_score > 0 else 0
                if movie_id not in movie_scores:
                    movie_scores[movie_id] = {'content': 0, 'collab': 0, 'movie': row}
                movie_scores[movie_id]['collab'] = score
        
        # Tính hybrid score
        results = []
        for movie_id, scores in movie_scores.items():
            hybrid_score = alpha * scores['content'] + (1 - alpha) * scores['collab']
            movie = scores['movie'].copy()
            movie['hybrid_score'] = hybrid_score
            movie['content_score'] = scores['content']
            movie['collab_score'] = scores['collab']
            results.append(movie)
        
        if len(results) == 0:
            return pd.DataFrame()
        
        # Tạo DataFrame và sắp xếp
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('hybrid_score', ascending=False)
        
        return results_df.head(n)
    
    def get_similar_movies(self, movie_id, n=10):
        """
        Tìm phim tương tự (dùng content-based vì không có user context)
        
        Args:
            movie_id: ID của phim
            n: Số lượng recommendations
        
        Returns:
            DataFrame với similar movies
        """
        return self.content_recommender.get_similar_movies(movie_id, n=n)
    
    def get_popular_movies(self, n=20):
        """Lấy popular movies (cold start)"""
        return self.content_recommender.get_popular_movies(limit=n)
    
    def get_trending_movies(self, n=20):
        """Lấy trending movies (cold start)"""
        from search_engine import MovieSearchEngine
        search_engine = MovieSearchEngine()
        return search_engine.get_trending_movies(limit=n)

def main():
    """Test hybrid recommender"""
    print("\n" + "=" * 80)
    print("TESTING HYBRID RECOMMENDER")
    print("=" * 80)
    
    # Khởi tạo recommender
    recommender = HybridRecommender()
    
    # Test 1: Cold start (chưa có ratings)
    print("\n1. Cold start recommendations (no ratings)...")
    recommendations = recommender.recommend(user_rated_movies={}, n=10)
    if len(recommendations) > 0:
        print(f"   Found {len(recommendations)} recommendations:")
        for idx, row in recommendations.head(5).iterrows():
            print(f"   - {row['title_clean']} ({row['year']:.0f})")
    
    # Test 2: Ít ratings (1-2 phim)
    print("\n2. Recommendations with few ratings (1 rating)...")
    user_history = {1: 5.0}  # Chỉ rate Toy Story
    recommendations = recommender.recommend(user_history, n=10)
    if len(recommendations) > 0:
        print(f"   Found {len(recommendations)} recommendations:")
        for idx, row in recommendations.head(5).iterrows():
            print(f"   - {row['title_clean']} ({row['year']:.0f}) - Hybrid: {row['hybrid_score']:.3f} "
                  f"(Content: {row['content_score']:.3f}, Collab: {row['collab_score']:.3f})")
    
    # Test 3: Nhiều ratings (5+ phim)
    print("\n3. Recommendations with many ratings (5 ratings)...")
    user_history = {1: 5.0, 2: 4.0, 3: 4.5, 4: 3.5, 5: 4.0}
    recommendations = recommender.recommend(user_history, n=10)
    if len(recommendations) > 0:
        print(f"   Found {len(recommendations)} recommendations:")
        for idx, row in recommendations.head(5).iterrows():
            print(f"   - {row['title_clean']} ({row['year']:.0f}) - Hybrid: {row['hybrid_score']:.3f} "
                  f"(Content: {row['content_score']:.3f}, Collab: {row['collab_score']:.3f})")
    
    # Test 4: Similar movies
    print("\n4. Similar movies to 'Toy Story' (movieId=1)...")
    similar = recommender.get_similar_movies(movie_id=1, n=5)
    if len(similar) > 0:
        print(f"   Found {len(similar)} similar movies:")
        for idx, row in similar.iterrows():
            print(f"   - {row['title_clean']} ({row['year']:.0f}) - Similarity: {row['similarity_score']:.3f}")
    
    print("\n" + "=" * 80)
    print("✅ Hybrid Recommender Test Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

