"""
Search Engine - Tìm kiếm phim theo title
Sử dụng fuzzy matching để tìm phim khớp với từ khóa
"""

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
import os
from pathlib import Path

# Đường dẫn - relative từ project root
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = str(BASE_DIR / "data_cleaned")

class MovieSearchEngine:
    """Search engine để tìm phim theo title"""
    
    def __init__(self):
        """Load dữ liệu movies"""
        self.movies = pd.read_csv(os.path.join(DATA_DIR, "movies_cleaned.csv"))
        print(f"✅ Loaded {len(self.movies)} movies for search")
    
    def search(self, query, limit=20, min_score=60):
        """
        Tìm kiếm phim theo query - Tên chuẩn nhất lên trước, na ná đằng sau
        
        Args:
            query: Từ khóa tìm kiếm
            limit: Số lượng kết quả tối đa
            min_score: Điểm tối thiểu để hiển thị (0-100)
        
        Returns:
            DataFrame với các phim khớp (sắp xếp: exact match → fuzzy match)
        """
        if not query or len(query.strip()) == 0:
            return pd.DataFrame()
        
        query = query.strip().lower()
        titles = self.movies['title_clean'].tolist()
        
        # 1. Tìm EXACT MATCH trước (contains hoặc exact)
        exact_matches = []
        fuzzy_matches = []
        
        # Tìm exact/contains matches
        for idx, title in enumerate(titles):
            title_lower = title.lower()
            # Exact match
            if title_lower == query:
                movie = self.movies.iloc[idx].copy()
                movie['match_score'] = 100  # Score cao nhất cho exact match
                movie['match_type'] = 'exact'
                exact_matches.append(movie)
            # Starts with
            elif title_lower.startswith(query):
                movie = self.movies.iloc[idx].copy()
                movie['match_score'] = 95  # Score cao cho starts with
                movie['match_type'] = 'starts_with'
                exact_matches.append(movie)
            # Contains
            elif query in title_lower:
                movie = self.movies.iloc[idx].copy()
                movie['match_score'] = 90  # Score cao cho contains
                movie['match_type'] = 'contains'
                exact_matches.append(movie)
        
        # 2. Tìm FUZZY MATCH (nếu chưa đủ limit)
        if len(exact_matches) < limit:
            # Tìm top matches với fuzzywuzzy
            all_matches = process.extract(
                query, 
                titles, 
                limit=limit * 3,  # Lấy nhiều hơn để filter
                scorer=fuzz.partial_ratio
            )
            
            # Lọc ra những cái chưa có trong exact_matches
            exact_titles = {m['title_clean'] for m in exact_matches}
            
            for title, score in all_matches:
                if score >= min_score and title not in exact_titles:
                    movie_idx = titles.index(title)
                    movie = self.movies.iloc[movie_idx].copy()
                    movie['match_score'] = score
                    movie['match_type'] = 'fuzzy'
                    fuzzy_matches.append(movie)
                    if len(exact_matches) + len(fuzzy_matches) >= limit:
                        break
        
        # 3. Kết hợp và sắp xếp: Exact matches trước, sau đó fuzzy matches
        all_results = exact_matches + fuzzy_matches
        
        if len(all_results) == 0:
            return pd.DataFrame()
        
        # Tạo DataFrame và sắp xếp: exact match (score cao) → fuzzy match (theo score)
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('match_score', ascending=False)
        
        # Giới hạn số lượng
        return results_df.head(limit)
    
    def search_exact(self, query, limit=20):
        """
        Tìm kiếm chính xác (contains) - nhanh hơn fuzzy
        """
        if not query or len(query.strip()) == 0:
            return pd.DataFrame()
        
        query = query.strip().lower()
        
        # Tìm kiếm contains (case-insensitive)
        mask = self.movies['title_clean'].str.lower().str.contains(query, na=False)
        results = self.movies[mask].copy()
        
        # Sắp xếp theo số ratings (phim phổ biến trước)
        results = results.sort_values('num_ratings', ascending=False)
        
        return results.head(limit)
    
    def get_movie_by_id(self, movie_id):
        """Lấy thông tin phim theo movieId"""
        movie = self.movies[self.movies['movieId'] == movie_id]
        if len(movie) > 0:
            return movie.iloc[0]
        return None
    
    def get_popular_movies(self, limit=20):
        """Lấy top phim phổ biến (nhiều ratings nhất)"""
        popular = self.movies.nlargest(limit, 'num_ratings')
        return popular
    
    def get_trending_movies(self, limit=20, min_ratings=10):
        """
        Lấy trending movies (rating cao và có nhiều ratings)
        Weighted score = (avg_rating * num_ratings) / total_ratings
        """
        # Lọc phim có ít nhất min_ratings
        filtered = self.movies[self.movies['num_ratings'] >= min_ratings].copy()
        
        # Tính weighted score
        total_ratings = filtered['num_ratings'].sum()
        filtered['trending_score'] = (
            filtered['avg_rating'] * filtered['num_ratings'] / total_ratings
        )
        
        # Sắp xếp và lấy top
        trending = filtered.nlargest(limit, 'trending_score')
        return trending[['movieId', 'title', 'title_clean', 'genres', 'year', 
                        'avg_rating', 'num_ratings', 'trending_score']]

def main():
    """Test search engine"""
    print("=" * 80)
    print("TESTING SEARCH ENGINE")
    print("=" * 80)
    
    # Khởi tạo search engine
    search_engine = MovieSearchEngine()
    
    # Test search
    print("\n1. Testing search with 'Avatar'...")
    results = search_engine.search("Avatar", limit=5)
    if len(results) > 0:
        print(f"   Found {len(results)} results:")
        for idx, row in results.iterrows():
            print(f"   - {row['title_clean']} ({row['year']:.0f}) - Score: {row['match_score']}")
    else:
        print("   No results found")
    
    print("\n2. Testing exact search with 'Toy'...")
    results = search_engine.search_exact("Toy", limit=5)
    if len(results) > 0:
        print(f"   Found {len(results)} results:")
        for idx, row in results.iterrows():
            print(f"   - {row['title_clean']} ({row['year']:.0f}) - Ratings: {row['num_ratings']}")
    
    print("\n3. Testing popular movies...")
    popular = search_engine.get_popular_movies(limit=5)
    print("   Top 5 popular movies:")
    for idx, row in popular.iterrows():
        print(f"   - {row['title_clean']} - {row['num_ratings']} ratings")
    
    print("\n4. Testing trending movies...")
    trending = search_engine.get_trending_movies(limit=5)
    print("   Top 5 trending movies:")
    for idx, row in trending.iterrows():
        print(f"   - {row['title_clean']} - Rating: {row['avg_rating']:.2f}, Ratings: {row['num_ratings']}")
    
    print("\n" + "=" * 80)
    print("✅ Search Engine Test Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

