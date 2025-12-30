"""
Script làm sạch và chuẩn hóa dữ liệu MovieLens
Thực hiện các tác vụ:
1. Xử lý missing values
2. Chuẩn hóa dữ liệu (tách year, genres, datetime)
3. Loại bỏ duplicates
4. Chuẩn bị vector hóa (TF-IDF)
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import os
from pathlib import Path

# Đường dẫn - relative từ project root
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = str(BASE_DIR / "data")
OUTPUT_DIR = str(BASE_DIR / "data_cleaned")

# Tạo thư mục output nếu chưa có
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_data():
    """Load dữ liệu gốc"""
    print("=" * 80)
    print("LOADING DATA...")
    print("=" * 80)
    
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    tags = pd.read_csv(os.path.join(DATA_DIR, "tags.csv"))
    links = pd.read_csv(os.path.join(DATA_DIR, "links.csv"))
    
    print("✅ Đã load thành công!")
    return movies, ratings, tags, links

def clean_movies(movies):
    """
    Làm sạch movies.csv
    1. Xử lý missing genres
    2. Tách year từ title
    3. Tách genres thành list
    """
    print("\n" + "=" * 80)
    print("LÀM SẠCH MOVIES.CSV")
    print("=" * 80)
    
    movies_clean = movies.copy()
    
    # 1. Xử lý missing genres
    print("\n1. Xử lý missing genres...")
    missing_genres_before = (movies_clean['genres'].isna() | 
                             (movies_clean['genres'] == '(no genres listed)')).sum()
    print(f"   - Số phim không có genres trước khi xử lý: {missing_genres_before}")
    
    # Gán "Unknown" cho genres rỗng
    movies_clean['genres'] = movies_clean['genres'].fillna('Unknown')
    movies_clean.loc[movies_clean['genres'] == '(no genres listed)', 'genres'] = 'Unknown'
    
    missing_genres_after = (movies_clean['genres'] == 'Unknown').sum()
    print(f"   - Số phim có genres 'Unknown' sau khi xử lý: {missing_genres_after}")
    
    # 2. Tách year từ title
    print("\n2. Tách year từ title...")
    def extract_year(title):
        """Tách năm từ title, ví dụ: 'Toy Story (1995)' -> 1995"""
        match = re.search(r'\((\d{4})\)', str(title))
        if match:
            year = int(match.group(1))
            # Kiểm tra năm hợp lý (1900-2025)
            if 1900 <= year <= 2025:
                return year
        return None
    
    movies_clean['year'] = movies_clean['title'].apply(extract_year)
    movies_with_year = movies_clean['year'].notna().sum()
    print(f"   - Số phim có year: {movies_with_year}/{len(movies_clean)}")
    print(f"   - Year min: {movies_clean['year'].min()}")
    print(f"   - Year max: {movies_clean['year'].max()}")
    
    # 3. Tách genres thành list
    print("\n3. Tách genres thành list...")
    movies_clean['genres_list'] = movies_clean['genres'].apply(
        lambda x: x.split('|') if pd.notna(x) and x != 'Unknown' else ['Unknown']
    )
    print(f"   - Sample genres_list: {movies_clean['genres_list'].iloc[0]}")
    
    # 4. Tạo title_clean (bỏ year trong title)
    print("\n4. Tạo title_clean (bỏ year)...")
    movies_clean['title_clean'] = movies_clean['title'].apply(
        lambda x: re.sub(r'\s*\(\d{4}\)\s*$', '', str(x)).strip()
    )
    print(f"   - Sample title: '{movies_clean['title'].iloc[0]}'")
    print(f"   - Sample title_clean: '{movies_clean['title_clean'].iloc[0]}'")
    
    print("\n✅ Hoàn thành làm sạch movies!")
    return movies_clean

def clean_ratings(ratings):
    """
    Làm sạch ratings.csv
    1. Loại bỏ duplicates (giữ bản cuối)
    2. Chuyển timestamp → datetime
    3. Tạo các features thời gian
    """
    print("\n" + "=" * 80)
    print("LÀM SẠCH RATINGS.CSV")
    print("=" * 80)
    
    ratings_clean = ratings.copy()
    
    # 1. Loại bỏ duplicates
    print("\n1. Kiểm tra và loại bỏ duplicates...")
    duplicates_before = ratings_clean.duplicated(subset=['userId', 'movieId']).sum()
    print(f"   - Số duplicates trước khi xử lý: {duplicates_before}")
    
    if duplicates_before > 0:
        # Giữ bản ghi cuối cùng (timestamp lớn nhất)
        ratings_clean = ratings_clean.sort_values('timestamp')
        ratings_clean = ratings_clean.drop_duplicates(
            subset=['userId', 'movieId'], 
            keep='last'
        )
        print(f"   - Đã loại bỏ {duplicates_before} duplicates")
    else:
        print("   - Không có duplicates")
    
    # 2. Chuyển timestamp → datetime
    print("\n2. Chuyển timestamp → datetime...")
    ratings_clean['datetime'] = pd.to_datetime(ratings_clean['timestamp'], unit='s')
    print(f"   - Ngày sớm nhất: {ratings_clean['datetime'].min()}")
    print(f"   - Ngày muộn nhất: {ratings_clean['datetime'].max()}")
    
    # 3. Tạo features thời gian
    print("\n3. Tạo features thời gian...")
    ratings_clean['year'] = ratings_clean['datetime'].dt.year
    ratings_clean['month'] = ratings_clean['datetime'].dt.month
    ratings_clean['day_of_week'] = ratings_clean['datetime'].dt.dayofweek
    print(f"   - Đã tạo: year, month, day_of_week")
    
    print("\n✅ Hoàn thành làm sạch ratings!")
    return ratings_clean

def clean_tags(tags):
    """
    Làm sạch tags.csv
    1. Xử lý missing tags
    2. Chuẩn hóa text (lower, strip)
    3. Chuyển timestamp → datetime
    """
    print("\n" + "=" * 80)
    print("LÀM SẠCH TAGS.CSV")
    print("=" * 80)
    
    tags_clean = tags.copy()
    
    # 1. Xử lý missing tags
    print("\n1. Xử lý missing tags...")
    missing_before = tags_clean['tag'].isna().sum()
    print(f"   - Số tags missing trước khi xử lý: {missing_before}")
    
    if missing_before > 0:
        # Bỏ các dòng có tag rỗng
        tags_clean = tags_clean.dropna(subset=['tag'])
        print(f"   - Đã bỏ {missing_before} dòng có tag rỗng")
    
    # 2. Chuẩn hóa text
    print("\n2. Chuẩn hóa text (lower, strip)...")
    tags_clean['tag'] = tags_clean['tag'].astype(str).str.lower().str.strip()
    print(f"   - Sample tag sau khi chuẩn hóa: '{tags_clean['tag'].iloc[0]}'")
    
    # 3. Chuyển timestamp → datetime
    print("\n3. Chuyển timestamp → datetime...")
    tags_clean['datetime'] = pd.to_datetime(tags_clean['timestamp'], unit='s')
    print(f"   - Ngày sớm nhất: {tags_clean['datetime'].min()}")
    print(f"   - Ngày muộn nhất: {tags_clean['datetime'].max()}")
    
    print("\n✅ Hoàn thành làm sạch tags!")
    return tags_clean

def aggregate_tags(tags_clean, movies_clean):
    """
    Aggregate tags theo movieId để tạo features cho content-based
    """
    print("\n" + "=" * 80)
    print("AGGREGATE TAGS THEO MOVIE")
    print("=" * 80)
    
    # Group tags theo movieId
    movie_tags = tags_clean.groupby('movieId')['tag'].apply(
        lambda x: ' '.join(x.unique())
    ).reset_index()
    movie_tags.columns = ['movieId', 'tags_combined']
    
    # Merge với movies
    movies_with_tags = movies_clean.merge(movie_tags, on='movieId', how='left')
    movies_with_tags['tags_combined'] = movies_with_tags['tags_combined'].fillna('')
    
    print(f"   - Số phim có tags: {movies_with_tags['tags_combined'].str.len().gt(0).sum()}")
    print(f"   - Sample tags_combined: '{movies_with_tags[movies_with_tags['tags_combined'].str.len() > 0]['tags_combined'].iloc[0][:100]}...'")
    
    return movies_with_tags

def prepare_content_features(movies_with_tags):
    """
    Chuẩn bị features cho content-based recommendation
    Tạo text kết hợp: title + genres + tags
    """
    print("\n" + "=" * 80)
    print("CHUẨN BỊ CONTENT FEATURES")
    print("=" * 80)
    
    movies_features = movies_with_tags.copy()
    
    # Tạo text kết hợp cho TF-IDF
    print("\n1. Tạo text kết hợp (title + genres + tags)...")
    
    def combine_features(row):
        """Kết hợp title, genres, tags thành một text"""
        title = str(row['title_clean']) if pd.notna(row['title_clean']) else ''
        genres = ' '.join(row['genres_list']) if isinstance(row['genres_list'], list) else str(row['genres'])
        tags = str(row['tags_combined']) if pd.notna(row['tags_combined']) else ''
        
        # Kết hợp với khoảng trắng
        combined = f"{title} {genres} {tags}".strip()
        return combined
    
    movies_features['content_text'] = movies_features.apply(combine_features, axis=1)
    
    print(f"   - Sample content_text: '{movies_features['content_text'].iloc[0][:150]}...'")
    print(f"   - Độ dài trung bình: {movies_features['content_text'].str.len().mean():.0f} ký tự")
    
    print("\n✅ Hoàn thành chuẩn bị content features!")
    return movies_features

def calculate_movie_stats(ratings_clean, movies_features):
    """
    Tính thống kê cho mỗi phim (avg_rating, num_ratings)
    """
    print("\n" + "=" * 80)
    print("TÍNH THỐNG KÊ MOVIES")
    print("=" * 80)
    
    # Tính average rating và số lượng ratings cho mỗi phim
    movie_stats = ratings_clean.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    
    movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings']
    
    # Merge với movies
    movies_with_stats = movies_features.merge(movie_stats, on='movieId', how='left')
    movies_with_stats['avg_rating'] = movies_with_stats['avg_rating'].fillna(0)
    movies_with_stats['num_ratings'] = movies_with_stats['num_ratings'].fillna(0).astype(int)
    
    print(f"   - Số phim có rating: {movies_with_stats['num_ratings'].gt(0).sum()}")
    print(f"   - Rating trung bình: {movies_with_stats['avg_rating'].mean():.2f}")
    print(f"   - Số ratings trung bình/phim: {movies_with_stats['num_ratings'].mean():.2f}")
    
    return movies_with_stats

def save_cleaned_data(movies_final, ratings_clean, tags_clean, links):
    """Lưu dữ liệu đã làm sạch"""
    print("\n" + "=" * 80)
    print("LƯU DỮ LIỆU ĐÃ LÀM SẠCH")
    print("=" * 80)
    
    # Lưu movies
    movies_output = movies_final[[
        'movieId', 'title', 'title_clean', 'genres', 'genres_list', 
        'year', 'tags_combined', 'content_text', 'avg_rating', 'num_ratings'
    ]]
    movies_output.to_csv(os.path.join(OUTPUT_DIR, "movies_cleaned.csv"), index=False)
    print(f"✅ Đã lưu: {OUTPUT_DIR}/movies_cleaned.csv ({len(movies_output)} dòng)")
    
    # Lưu ratings
    ratings_output = ratings_clean[[
        'userId', 'movieId', 'rating', 'timestamp', 
        'datetime', 'year', 'month', 'day_of_week'
    ]]
    ratings_output.to_csv(os.path.join(OUTPUT_DIR, "ratings_cleaned.csv"), index=False)
    print(f"✅ Đã lưu: {OUTPUT_DIR}/ratings_cleaned.csv ({len(ratings_output)} dòng)")
    
    # Lưu tags
    tags_output = tags_clean[[
        'userId', 'movieId', 'tag', 'timestamp', 'datetime'
    ]]
    tags_output.to_csv(os.path.join(OUTPUT_DIR, "tags_cleaned.csv"), index=False)
    print(f"✅ Đã lưu: {OUTPUT_DIR}/tags_cleaned.csv ({len(tags_output)} dòng)")
    
    # Lưu links (không cần làm sạch nhiều)
    links.to_csv(os.path.join(OUTPUT_DIR, "links_cleaned.csv"), index=False)
    print(f"✅ Đã lưu: {OUTPUT_DIR}/links_cleaned.csv ({len(links)} dòng)")

def summary(movies_final, ratings_clean, tags_clean):
    """Tóm tắt kết quả làm sạch"""
    print("\n" + "=" * 80)
    print("TÓM TẮT KẾT QUẢ LÀM SẠCH")
    print("=" * 80)
    
    print("\n✅ ĐÃ HOÀN THÀNH:")
    print("   1. ✅ Xử lý missing values (genres → 'Unknown')")
    print("   2. ✅ Chuẩn hóa dữ liệu:")
    print("      - Tách year từ title")
    print("      - Tách genres thành list")
    print("      - Chuyển timestamp → datetime")
    print("      - Chuẩn hóa text (lower, strip)")
    print("   3. ✅ Loại bỏ duplicates (nếu có)")
    print("   4. ✅ Aggregate tags theo movie")
    print("   5. ✅ Tạo content_text cho TF-IDF")
    print("   6. ✅ Tính thống kê movies (avg_rating, num_ratings)")
    
    print("\n📊 THỐNG KÊ SAU KHI LÀM SẠCH:")
    print(f"   - Movies: {len(movies_final):,} phim")
    print(f"   - Ratings: {len(ratings_clean):,} ratings")
    print(f"   - Tags: {len(tags_clean):,} tags")
    print(f"   - Phim có genres 'Unknown': {(movies_final['genres'] == 'Unknown').sum()}")
    print(f"   - Phim có year: {movies_final['year'].notna().sum()}")
    print(f"   - Phim có tags: {movies_final['tags_combined'].str.len().gt(0).sum()}")
    print(f"   - Phim có rating: {movies_final['num_ratings'].gt(0).sum()}")
    
    print("\n📋 CÁC BƯỚC TIẾP THEO:")
    print("   1. Trực quan hóa dữ liệu (visualization.py)")
    print("   2. Vector hóa với TF-IDF (sẽ làm trong model)")
    print("   3. Xây dựng recommendation models")

def main():
    """Hàm chính"""
    print("\n" + "=" * 80)
    print("LÀM SẠCH DỮ LIỆU MOVIELENS")
    print("=" * 80)
    
    # Load data
    movies, ratings, tags, links = load_data()
    
    # Làm sạch từng file
    movies_clean = clean_movies(movies)
    ratings_clean = clean_ratings(ratings)
    tags_clean = clean_tags(tags)
    
    # Aggregate tags
    movies_with_tags = aggregate_tags(tags_clean, movies_clean)
    
    # Chuẩn bị content features
    movies_features = prepare_content_features(movies_with_tags)
    
    # Tính thống kê
    movies_final = calculate_movie_stats(ratings_clean, movies_features)
    
    # Lưu dữ liệu đã làm sạch
    save_cleaned_data(movies_final, ratings_clean, tags_clean, links)
    
    # Tóm tắt
    summary(movies_final, ratings_clean, tags_clean)
    
    print("\n" + "=" * 80)
    print("HOÀN THÀNH LÀM SẠCH DỮ LIỆU!")
    print("=" * 80)
    
    return movies_final, ratings_clean, tags_clean

if __name__ == "__main__":
    main()

