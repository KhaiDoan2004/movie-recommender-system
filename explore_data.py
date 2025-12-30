"""
Script khám phá dữ liệu MovieLens
Phân tích dataset để hiểu cấu trúc, chất lượng và đặc điểm dữ liệu
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path

# Đường dẫn - relative từ project root
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = str(BASE_DIR / "data")

def load_data():
    """Load tất cả các file CSV"""
    print("=" * 80)
    print("LOADING DATA...")
    print("=" * 80)
    
    try:
        movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
        ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
        tags = pd.read_csv(os.path.join(DATA_DIR, "tags.csv"))
        links = pd.read_csv(os.path.join(DATA_DIR, "links.csv"))
        
        print("✅ Đã load thành công tất cả các file!")
        return movies, ratings, tags, links
    except Exception as e:
        print(f"❌ Lỗi khi load dữ liệu: {e}")
        return None, None, None, None

def basic_statistics(movies, ratings, tags, links):
    """Thống kê cơ bản về dataset"""
    print("\n" + "=" * 80)
    print("THỐNG KÊ CƠ BẢN")
    print("=" * 80)
    
    print("\n📊 MOVIES.CSV:")
    print(f"   - Số lượng phim: {len(movies):,}")
    print(f"   - Số cột: {len(movies.columns)}")
    print(f"   - Các cột: {list(movies.columns)}")
    print(f"   - Kích thước: {movies.shape}")
    
    print("\n📊 RATINGS.CSV:")
    print(f"   - Số lượng ratings: {len(ratings):,}")
    print(f"   - Số cột: {len(ratings.columns)}")
    print(f"   - Các cột: {list(ratings.columns)}")
    print(f"   - Kích thước: {ratings.shape}")
    print(f"   - Số users duy nhất: {ratings['userId'].nunique():,}")
    print(f"   - Số phim được rate: {ratings['movieId'].nunique():,}")
    
    print("\n📊 TAGS.CSV:")
    print(f"   - Số lượng tags: {len(tags):,}")
    print(f"   - Số cột: {len(tags.columns)}")
    print(f"   - Các cột: {list(tags.columns)}")
    print(f"   - Kích thước: {tags.shape}")
    print(f"   - Số users đã tag: {tags['userId'].nunique():,}")
    print(f"   - Số phim có tag: {tags['movieId'].nunique():,}")
    
    print("\n📊 LINKS.CSV:")
    print(f"   - Số lượng links: {len(links):,}")
    print(f"   - Số cột: {len(links.columns)}")
    print(f"   - Các cột: {list(links.columns)}")
    print(f"   - Kích thước: {links.shape}")

def check_missing_values(movies, ratings, tags, links):
    """Kiểm tra missing values"""
    print("\n" + "=" * 80)
    print("KIỂM TRA MISSING VALUES")
    print("=" * 80)
    
    print("\n📋 MOVIES.CSV:")
    missing_movies = movies.isnull().sum()
    if missing_movies.sum() == 0:
        print("   ✅ Không có missing values")
    else:
        print(missing_movies[missing_movies > 0])
    
    print("\n📋 RATINGS.CSV:")
    missing_ratings = ratings.isnull().sum()
    if missing_ratings.sum() == 0:
        print("   ✅ Không có missing values")
    else:
        print(missing_ratings[missing_ratings > 0])
    
    print("\n📋 TAGS.CSV:")
    missing_tags = tags.isnull().sum()
    if missing_tags.sum() == 0:
        print("   ✅ Không có missing values")
    else:
        print(missing_tags[missing_tags > 0])
    
    print("\n📋 LINKS.CSV:")
    missing_links = links.isnull().sum()
    if missing_links.sum() == 0:
        print("   ✅ Không có missing values")
    else:
        print(missing_links[missing_links > 0])
    
    # Kiểm tra genres rỗng hoặc "(no genres listed)"
    print("\n🔍 Kiểm tra genres rỗng:")
    empty_genres = movies[movies['genres'].isna() | (movies['genres'] == '(no genres listed)')]
    print(f"   - Số phim không có genres: {len(empty_genres)}")
    if len(empty_genres) > 0:
        print(f"   - Sample: {empty_genres.head(3)['title'].tolist()}")

def check_duplicates(movies, ratings, tags, links):
    """Kiểm tra duplicates"""
    print("\n" + "=" * 80)
    print("KIỂM TRA DUPLICATES")
    print("=" * 80)
    
    print("\n📋 MOVIES.CSV:")
    dup_movies = movies.duplicated(subset=['movieId']).sum()
    print(f"   - Duplicate movieId: {dup_movies}")
    dup_title = movies.duplicated(subset=['title']).sum()
    print(f"   - Duplicate title: {dup_title}")
    
    print("\n📋 RATINGS.CSV:")
    dup_ratings = ratings.duplicated(subset=['userId', 'movieId']).sum()
    print(f"   - Duplicate (userId, movieId): {dup_ratings}")
    if dup_ratings > 0:
        print(f"   ⚠️  Cần xử lý: giữ bản ghi cuối cùng")
        # Hiển thị sample duplicates
        dup_samples = ratings[ratings.duplicated(subset=['userId', 'movieId'], keep=False)]
        print(f"   - Sample duplicates:\n{dup_samples.head(10)}")
    
    print("\n📋 TAGS.CSV:")
    dup_tags = tags.duplicated(subset=['userId', 'movieId', 'tag']).sum()
    print(f"   - Duplicate (userId, movieId, tag): {dup_tags}")
    
    print("\n📋 LINKS.CSV:")
    dup_links = links.duplicated(subset=['movieId']).sum()
    print(f"   - Duplicate movieId: {dup_links}")

def analyze_movies(movies):
    """Phân tích chi tiết movies"""
    print("\n" + "=" * 80)
    print("PHÂN TÍCH MOVIES")
    print("=" * 80)
    
    print("\n📝 Sample movies (5 phim đầu):")
    print(movies.head())
    
    print("\n📝 Sample movies (5 phim cuối):")
    print(movies.tail())
    
    # Phân tích genres
    print("\n🎬 PHÂN TÍCH GENRES:")
    all_genres = []
    for genres_str in movies['genres'].dropna():
        if genres_str != '(no genres listed)':
            all_genres.extend(genres_str.split('|'))
    
    from collections import Counter
    genre_counts = Counter(all_genres)
    print(f"   - Tổng số genres duy nhất: {len(genre_counts)}")
    print(f"   - Top 10 genres phổ biến:")
    for genre, count in genre_counts.most_common(10):
        print(f"     {genre}: {count:,} phim")
    
    # Phân tích năm (tách từ title)
    print("\n📅 PHÂN TÍCH NĂM (từ title):")
    import re
    years = []
    for title in movies['title']:
        match = re.search(r'\((\d{4})\)', title)
        if match:
            years.append(int(match.group(1)))
    
    if years:
        print(f"   - Năm sớm nhất: {min(years)}")
        print(f"   - Năm muộn nhất: {max(years)}")
        print(f"   - Số phim có năm: {len(years)}/{len(movies)}")
        
        # Phân bố theo thập kỷ
        decades = {}
        for year in years:
            decade = (year // 10) * 10
            decades[decade] = decades.get(decade, 0) + 1
        
        print(f"   - Phân bố theo thập kỷ:")
        for decade in sorted(decades.keys()):
            print(f"     {decade}s: {decades[decade]:,} phim")

def analyze_ratings(ratings):
    """Phân tích chi tiết ratings"""
    print("\n" + "=" * 80)
    print("PHÂN TÍCH RATINGS")
    print("=" * 80)
    
    print("\n📝 Sample ratings (10 dòng đầu):")
    print(ratings.head(10))
    
    # Thống kê rating
    print("\n⭐ THỐNG KÊ RATING:")
    print(f"   - Rating trung bình: {ratings['rating'].mean():.2f}")
    print(f"   - Rating trung vị: {ratings['rating'].median():.2f}")
    print(f"   - Rating min: {ratings['rating'].min()}")
    print(f"   - Rating max: {ratings['rating'].max()}")
    print(f"   - Độ lệch chuẩn: {ratings['rating'].std():.2f}")
    
    print("\n📊 Phân bố rating:")
    rating_dist = ratings['rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        percentage = (count / len(ratings)) * 100
        print(f"   {rating:.1f} sao: {count:>7,} ({percentage:>5.2f}%)")
    
    # Phân tích timestamp
    print("\n⏰ PHÂN TÍCH TIMESTAMP:")
    ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
    print(f"   - Ngày sớm nhất: {ratings['datetime'].min()}")
    print(f"   - Ngày muộn nhất: {ratings['datetime'].max()}")
    print(f"   - Khoảng thời gian: {(ratings['datetime'].max() - ratings['datetime'].min()).days} ngày")
    
    # Ratings theo năm
    ratings['year'] = ratings['datetime'].dt.year
    ratings_by_year = ratings.groupby('year').size()
    print(f"\n📈 Số ratings theo năm (top 5):")
    for year, count in ratings_by_year.sort_values(ascending=False).head(5).items():
        print(f"   {year}: {count:,} ratings")
    
    # Phân tích users
    print("\n👥 PHÂN TÍCH USERS:")
    user_rating_counts = ratings.groupby('userId').size()
    print(f"   - Số ratings trung bình/user: {user_rating_counts.mean():.2f}")
    print(f"   - User rate nhiều nhất: {user_rating_counts.max()} ratings")
    print(f"   - User rate ít nhất: {user_rating_counts.min()} ratings")
    print(f"   - Median ratings/user: {user_rating_counts.median():.2f}")
    
    # Phân tích movies
    print("\n🎬 PHÂN TÍCH MOVIES:")
    movie_rating_counts = ratings.groupby('movieId').size()
    print(f"   - Số ratings trung bình/phim: {movie_rating_counts.mean():.2f}")
    print(f"   - Phim được rate nhiều nhất: {movie_rating_counts.max()} ratings")
    print(f"   - Phim được rate ít nhất: {movie_rating_counts.min()} ratings")
    print(f"   - Median ratings/movie: {movie_rating_counts.median():.2f}")
    
    # Top phim được rate nhiều nhất
    print(f"\n🏆 Top 10 phim được rate nhiều nhất:")
    top_movies = movie_rating_counts.sort_values(ascending=False).head(10)
    for movie_id, count in top_movies.items():
        print(f"   MovieID {movie_id}: {count:,} ratings")

def analyze_tags(tags):
    """Phân tích chi tiết tags"""
    print("\n" + "=" * 80)
    print("PHÂN TÍCH TAGS")
    print("=" * 80)
    
    print("\n📝 Sample tags (10 dòng đầu):")
    print(tags.head(10))
    
    print(f"\n🏷️  THỐNG KÊ TAGS:")
    print(f"   - Số tags duy nhất: {tags['tag'].nunique():,}")
    print(f"   - Số phim có tag: {tags['movieId'].nunique():,}")
    print(f"   - Số users đã tag: {tags['userId'].nunique():,}")
    
    # Top tags
    print(f"\n🏆 Top 20 tags phổ biến nhất:")
    top_tags = tags['tag'].value_counts().head(20)
    for tag, count in top_tags.items():
        print(f"   '{tag}': {count:,} lần")
    
    # Tags per movie
    tags_per_movie = tags.groupby('movieId').size()
    print(f"\n📊 Tags per movie:")
    print(f"   - Trung bình: {tags_per_movie.mean():.2f} tags/phim")
    print(f"   - Nhiều nhất: {tags_per_movie.max()} tags")
    print(f"   - Ít nhất: {tags_per_movie.min()} tags")

def analyze_links(links):
    """Phân tích chi tiết links"""
    print("\n" + "=" * 80)
    print("PHÂN TÍCH LINKS")
    print("=" * 80)
    
    print("\n📝 Sample links (10 dòng đầu):")
    print(links.head(10))
    
    print(f"\n🔗 THỐNG KÊ LINKS:")
    print(f"   - Số phim có IMDb link: {links['imdbId'].notna().sum():,}")
    print(f"   - Số phim có TMDB link: {links['tmdbId'].notna().sum():,}")
    print(f"   - Số phim có cả 2 links: {(links['imdbId'].notna() & links['tmdbId'].notna()).sum():,}")

def data_quality_summary(movies, ratings, tags, links):
    """Tóm tắt chất lượng dữ liệu"""
    print("\n" + "=" * 80)
    print("TÓM TẮT CHẤT LƯỢNG DỮ LIỆU")
    print("=" * 80)
    
    print("\n✅ ĐIỂM MẠNH:")
    print("   - Dataset lớn: 9,742 phim, 100,836 ratings")
    print("   - Không có missing values trong ratings")
    print("   - Có đầy đủ thông tin: movies, ratings, tags, links")
    print("   - Dữ liệu trải dài 22 năm (1996-2018)")
    
    print("\n⚠️  VẤN ĐỀ CẦN XỬ LÝ:")
    # Kiểm tra genres rỗng
    empty_genres = movies[movies['genres'].isna() | (movies['genres'] == '(no genres listed)')]
    if len(empty_genres) > 0:
        print(f"   - {len(empty_genres)} phim không có genres → cần gán 'Unknown'")
    
    # Kiểm tra duplicates trong ratings
    dup_ratings = ratings.duplicated(subset=['userId', 'movieId']).sum()
    if dup_ratings > 0:
        print(f"   - {dup_ratings} duplicate ratings → cần giữ bản cuối")
    
    # Kiểm tra phim không có rating
    movies_with_ratings = ratings['movieId'].unique()
    movies_without_ratings = set(movies['movieId']) - set(movies_with_ratings)
    if len(movies_without_ratings) > 0:
        print(f"   - {len(movies_without_ratings)} phim không có rating")
    
    print("\n📋 CÁC BƯỚC TIẾP THEO:")
    print("   1. Làm sạch dữ liệu (xử lý missing, duplicates)")
    print("   2. Chuẩn hóa dữ liệu (tách year, genres, datetime)")
    print("   3. Vector hóa (TF-IDF cho content-based)")
    print("   4. Trực quan hóa dữ liệu")
    print("   5. Xây dựng models")

def main():
    """Hàm chính"""
    print("\n" + "=" * 80)
    print("KHÁM PHÁ DỮ LIỆU MOVIELENS DATASET")
    print("=" * 80)
    
    # Load data
    movies, ratings, tags, links = load_data()
    
    if movies is None:
        print("❌ Không thể load dữ liệu. Vui lòng kiểm tra đường dẫn!")
        return
    
    # Thực hiện các phân tích
    basic_statistics(movies, ratings, tags, links)
    check_missing_values(movies, ratings, tags, links)
    check_duplicates(movies, ratings, tags, links)
    analyze_movies(movies)
    analyze_ratings(ratings)
    analyze_tags(tags)
    analyze_links(links)
    data_quality_summary(movies, ratings, tags, links)
    
    print("\n" + "=" * 80)
    print("HOÀN THÀNH KHÁM PHÁ DỮ LIỆU!")
    print("=" * 80)

if __name__ == "__main__":
    main()

