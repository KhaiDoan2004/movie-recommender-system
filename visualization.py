"""
Script trực quan hóa dữ liệu MovieLens
Tạo 10-12 biểu đồ (3 biểu đồ tương tác) và tổng hợp vào dashboard HTML
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Cấu hình - relative từ project root
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = str(BASE_DIR / "data_cleaned")
OUTPUT_DIR = str(BASE_DIR / "images")
DASHBOARD_DIR = str(BASE_DIR / "dashboard")

# Tạo thư mục
for dir_path in [OUTPUT_DIR, DASHBOARD_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Cấu hình style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load dữ liệu đã làm sạch"""
    print("=" * 80)
    print("LOADING DATA FOR VISUALIZATION...")
    print("=" * 80)
    
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies_cleaned.csv"))
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings_cleaned.csv"))
    tags = pd.read_csv(os.path.join(DATA_DIR, "tags_cleaned.csv"))
    
    # Chuyển datetime
    ratings['datetime'] = pd.to_datetime(ratings['datetime'])
    tags['datetime'] = pd.to_datetime(tags['datetime'])
    
    print(f"✅ Loaded {len(movies)} movies")
    print(f"✅ Loaded {len(ratings)} ratings")
    print(f"✅ Loaded {len(tags)} tags")
    
    return movies, ratings, tags

# ============================================================================
# BIỂU ĐỒ 1-8: STATIC (Matplotlib/Seaborn)
# ============================================================================

def plot1_rating_distribution(ratings):
    """1. Histogram phân bố rating"""
    print("\n📊 Creating plot 1: Rating Distribution...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rating_counts = ratings['rating'].value_counts().sort_index()
    
    # Tạo x positions rõ ràng, không trùng chéo
    x_pos = np.arange(len(rating_counts))
    width = 0.6  # Độ rộng cột
    
    # Tạo gradient màu từ xanh nhạt đến xanh đậm
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(rating_counts)))
    
    bars = ax.bar(x_pos, rating_counts.values, 
                  width=width,
                  color=colors, 
                  alpha=0.8, 
                  edgecolor='black',
                  linewidth=1.5)
    
    # Thêm số trên mỗi cột
    for i, (bar, (rating, count)) in enumerate(zip(bars, rating_counts.items())):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Thêm giá trị rating dưới cột
        ax.text(bar.get_x() + bar.get_width()/2., -max(rating_counts.values)*0.02,
                f'{rating:.1f}',
                ha='center', va='top', fontsize=10, fontweight='bold', color='#333')
    
    # Set x-axis labels với giá trị rating đầy đủ
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{r:.1f}' for r in rating_counts.index], fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Rating', fontsize=13, fontweight='bold', color='#333')
    ax.set_ylabel('Count', fontsize=13, fontweight='bold', color='#333')
    ax.set_title('Rating Distribution (0.5 - 5.0 stars)', fontsize=16, fontweight='bold', pad=20, color='#333')
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_rating_distribution.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("   ✅ Saved: 1_rating_distribution.png")
    return '1_rating_distribution.png'

def plot2_top_genres(movies):
    """2. Bar chart top genres"""
    print("\n📊 Creating plot 2: Top Genres...")
    
    # Đếm genres
    all_genres = []
    for genres_list in movies['genres_list']:
        if isinstance(genres_list, str):
            genres_list = eval(genres_list)
        if isinstance(genres_list, list):
            all_genres.extend(genres_list)
    
    from collections import Counter
    genre_counts = Counter(all_genres)
    top_genres = dict(genre_counts.most_common(15))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    genres = list(top_genres.keys())
    counts = list(top_genres.values())
    
    bars = ax.barh(genres, counts, color='coral', alpha=0.8, edgecolor='black')
    ax.invert_yaxis()
    
    # Thêm số
    for i, (genre, count) in enumerate(zip(genres, counts)):
        ax.text(count, i, f' {count:,}', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Number of Movies', fontsize=12, fontweight='bold')
    ax.set_ylabel('Genres', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Most Popular Genres', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_top_genres.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 2_top_genres.png")
    return '2_top_genres.png'

def plot3_top_rated_movies(movies):
    """3. Top 20 phim được rate nhiều nhất"""
    print("\n📊 Creating plot 3: Top Rated Movies (by count)...")
    
    top_movies = movies.nlargest(20, 'num_ratings')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(top_movies)), top_movies['num_ratings'].values,
                   color='mediumseagreen', alpha=0.8, edgecolor='black')
    ax.invert_yaxis()
    
    # Labels
    titles = [title[:40] + '...' if len(title) > 40 else title 
              for title in top_movies['title_clean'].values]
    ax.set_yticks(range(len(top_movies)))
    ax.set_yticklabels(titles, fontsize=9)
    
    # Thêm số
    for i, count in enumerate(top_movies['num_ratings'].values):
        ax.text(count, i, f' {count:,}', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Number of Ratings', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Most Rated Movies', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_top_rated_movies.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 3_top_rated_movies.png")
    return '3_top_rated_movies.png'

def plot4_highest_rated_movies(movies):
    """4. Top 20 phim rating cao nhất (ít nhất 50 ratings)"""
    print("\n📊 Creating plot 4: Highest Rated Movies...")
    
    # Lọc phim có ít nhất 50 ratings
    filtered = movies[movies['num_ratings'] >= 50]
    top_rated = filtered.nlargest(20, 'avg_rating')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(top_rated)), top_rated['avg_rating'].values,
                   color='gold', alpha=0.8, edgecolor='black')
    ax.invert_yaxis()
    
    # Labels
    titles = [title[:40] + '...' if len(title) > 40 else title 
              for title in top_rated['title_clean'].values]
    ax.set_yticks(range(len(top_rated)))
    ax.set_yticklabels(titles, fontsize=9)
    
    # Thêm số
    for i, rating in enumerate(top_rated['avg_rating'].values):
        ax.text(rating, i, f' {rating:.2f}', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Average Rating', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Highest Rated Movies (≥50 ratings)', fontsize=14, fontweight='bold')
    ax.set_xlim([4.0, 5.0])
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '4_highest_rated_movies.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 4_highest_rated_movies.png")
    return '4_highest_rated_movies.png'

def plot5_ratings_by_year(ratings):
    """5. Line chart số ratings theo năm"""
    print("\n📊 Creating plot 5: Ratings by Year...")
    
    ratings_by_year = ratings.groupby('year').size()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ratings_by_year.index, ratings_by_year.values, 
            marker='o', linewidth=2, markersize=8, color='purple', alpha=0.7)
    ax.fill_between(ratings_by_year.index, ratings_by_year.values, 
                     alpha=0.3, color='purple')
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Ratings', fontsize=12, fontweight='bold')
    ax.set_title('Ratings by Year (1996-2018)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ratings_by_year.index[::2])  # Hiển thị mỗi 2 năm
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '5_ratings_by_year.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 5_ratings_by_year.png")
    return '5_ratings_by_year.png'

def plot6_user_rating_distribution(ratings):
    """6. Histogram số ratings/user"""
    print("\n📊 Creating plot 6: User Rating Distribution...")
    
    user_rating_counts = ratings.groupby('userId').size()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(user_rating_counts.values, bins=50, color='teal', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Number of Ratings per User', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Users', fontsize=12, fontweight='bold')
    ax.set_title('User Rating Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Thêm thống kê
    median = user_rating_counts.median()
    mean = user_rating_counts.mean()
    ax.axvline(median, color='red', linestyle='--', linewidth=2, label=f'Median: {median:.0f}')
    ax.axvline(mean, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean:.0f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '6_user_rating_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 6_user_rating_distribution.png")
    return '6_user_rating_distribution.png'

def plot7_movie_rating_distribution(movies):
    """7. Histogram số ratings/movie"""
    print("\n📊 Creating plot 7: Movie Rating Distribution...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(movies[movies['num_ratings'] > 0]['num_ratings'].values, 
            bins=50, color='salmon', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Number of Ratings per Movie', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Movies', fontsize=12, fontweight='bold')
    ax.set_title('Movie Rating Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xscale('log')  # Log scale vì phân bố lệch
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '7_movie_rating_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 7_movie_rating_distribution.png")
    return '7_movie_rating_distribution.png'

def plot8_genre_correlation_heatmap(movies):
    """8. Heatmap correlation giữa genres"""
    print("\n📊 Creating plot 8: Genre Correlation Heatmap...")
    
    # Tạo matrix genres
    all_genres = set()
    for genres_list in movies['genres_list']:
        if isinstance(genres_list, str):
            genres_list = eval(genres_list)
        if isinstance(genres_list, list):
            all_genres.update(genres_list)
    
    all_genres = sorted(list(all_genres))
    genre_matrix = np.zeros((len(all_genres), len(all_genres)))
    
    for genres_list in movies['genres_list']:
        if isinstance(genres_list, str):
            genres_list = eval(genres_list)
        if isinstance(genres_list, list):
            for i, g1 in enumerate(all_genres):
                for j, g2 in enumerate(all_genres):
                    if g1 in genres_list and g2 in genres_list:
                        genre_matrix[i][j] += 1
    
    # Tính correlation
    genre_df = pd.DataFrame(genre_matrix, index=all_genres, columns=all_genres)
    # Chỉ lấy top 10 genres
    top_genres = genre_df.sum().nlargest(10).index
    genre_df_top = genre_df.loc[top_genres, top_genres]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(genre_df_top, annot=True, fmt='.0f', cmap='YlOrRd', 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Heatmap: Number of Movies with Both Genres (Top 10)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '8_genre_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 8_genre_correlation_heatmap.png")
    return '8_genre_correlation_heatmap.png'

# ============================================================================
# BIỂU ĐỒ 9-11: INTERACTIVE (Plotly)
# ============================================================================

def plot9_scatter_rating_vs_count(movies):
    """9. Scatter plot: avg_rating vs num_ratings (INTERACTIVE)"""
    print("\n📊 Creating plot 9: Rating vs Count Scatter (Interactive)...")
    
    # Lọc phim có ratings
    movies_with_ratings = movies[movies['num_ratings'] > 0].copy()
    
    fig = px.scatter(
        movies_with_ratings,
        x='num_ratings',
        y='avg_rating',
        size='num_ratings',
        color='avg_rating',
        hover_data=['title_clean', 'genres'],
        title='Scatter Plot: Average Rating vs Number of Ratings',
        labels={
            'num_ratings': 'Number of Ratings',
            'avg_rating': 'Average Rating',
            'title_clean': 'Movie Title',
            'genres': 'Genres'
        },
        color_continuous_scale='Viridis',
        size_max=20
    )
    
    fig.update_layout(
        width=1000,
        height=600,
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12
    )
    
    fig.write_html(os.path.join(OUTPUT_DIR, '9_scatter_rating_vs_count.html'))
    print("   ✅ Saved: 9_scatter_rating_vs_count.html")
    return '9_scatter_rating_vs_count.html'

def plot10_ratings_by_month(ratings):
    """10. Line chart ratings theo tháng (INTERACTIVE)"""
    print("\n📊 Creating plot 10: Ratings by Month (Interactive)...")
    
    ratings['year_month'] = ratings['datetime'].dt.to_period('M')
    ratings_by_month = ratings.groupby('year_month').size().reset_index()
    ratings_by_month['year_month'] = ratings_by_month['year_month'].astype(str)
    
    fig = px.line(
        ratings_by_month,
        x='year_month',
        y=0,
        title='Ratings by Month (1996-2018)',
        labels={'year_month': 'Month', 0: 'Number of Ratings'},
        markers=True
    )
    
    fig.update_layout(
        width=1200,
        height=500,
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12,
        xaxis_tickangle=-45
    )
    
    fig.write_html(os.path.join(OUTPUT_DIR, '10_ratings_by_month.html'))
    print("   ✅ Saved: 10_ratings_by_month.html")
    return '10_ratings_by_month.html'

def plot11_top_genres_by_year(movies, ratings):
    """11. Top genres theo năm (INTERACTIVE)"""
    print("\n📊 Creating plot 11: Top Genres by Year (Interactive)...")
    
    # Dùng year từ movies (năm phim được sản xuất), không merge với ratings
    movies_with_year = movies[movies['year'].notna()].copy()
    
    # Tách genres và đếm
    genre_year_data = []
    for _, row in movies_with_year.iterrows():
        year = row['year']
        if pd.notna(year):
            genres_list = row['genres_list']
            if isinstance(genres_list, str):
                genres_list = eval(genres_list)
            if isinstance(genres_list, list):
                for genre in genres_list:
                    genre_year_data.append({'year': int(year), 'genre': genre})
    
    genre_year_df = pd.DataFrame(genre_year_data)
    genre_year_counts = genre_year_df.groupby(['year', 'genre']).size().reset_index(name='count')
    
    # Top 5 genres mỗi năm
    top_genres_by_year = []
    for year in genre_year_counts['year'].unique():
        year_data = genre_year_counts[genre_year_counts['year'] == year]
        top_5 = year_data.nlargest(5, 'count')
        for _, row in top_5.iterrows():
            top_genres_by_year.append({
                'year': year,
                'genre': row['genre'],
                'count': row['count'],
                'rank': list(top_5['genre'].values).index(row['genre']) + 1
            })
    
    top_df = pd.DataFrame(top_genres_by_year)
    
    fig = px.bar(
        top_df,
        x='year',
        y='count',
        color='genre',
        title='Top 5 Genres by Year (Movie Release Year)',
        labels={'year': 'Year', 'count': 'Number of Movies', 'genre': 'Genre'},
        barmode='stack'
    )
    
    fig.update_layout(
        width=1200,
        height=600,
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12
    )
    
    fig.write_html(os.path.join(OUTPUT_DIR, '11_top_genres_by_year.html'))
    print("   ✅ Saved: 11_top_genres_by_year.html")
    return '11_top_genres_by_year.html'

def plot12_rating_by_genre_boxplot(movies):
    """12. Box plot phân bố rating theo genres"""
    print("\n📊 Creating plot 12: Rating Distribution by Genre...")
    
    # Merge với ratings để có rating
    movies_with_ratings = movies[movies['num_ratings'] > 0].copy()
    
    # Tách genres và tạo data
    genre_rating_data = []
    for _, row in movies_with_ratings.iterrows():
        avg_rating = row['avg_rating']
        genres_list = row['genres_list']
        if isinstance(genres_list, str):
            genres_list = eval(genres_list)
        if isinstance(genres_list, list):
            for genre in genres_list:
                genre_rating_data.append({'genre': genre, 'avg_rating': avg_rating})
    
    genre_rating_df = pd.DataFrame(genre_rating_data)
    
    # Top 10 genres
    top_genres = genre_rating_df.groupby('genre')['avg_rating'].count().nlargest(10).index
    genre_rating_top = genre_rating_df[genre_rating_df['genre'].isin(top_genres)]
    
    # Sắp xếp theo median rating để dễ so sánh
    genre_medians = genre_rating_top.groupby('genre')['avg_rating'].median().sort_values(ascending=False)
    genre_order = genre_medians.index.tolist()
    genre_rating_top['genre'] = pd.Categorical(genre_rating_top['genre'], categories=genre_order, ordered=True)
    genre_rating_top = genre_rating_top.sort_values('genre')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Tạo boxplot với seaborn để có màu sắc đẹp hơn
    box_plot = sns.boxplot(
        data=genre_rating_top,
        x='genre',
        y='avg_rating',
        palette='viridis',  # Gradient màu từ xanh đến vàng
        ax=ax,
        linewidth=2,
        fliersize=4
    )
    
    # Tùy chỉnh màu sắc cho từng box
    colors = plt.cm.viridis(np.linspace(0, 1, len(genre_order)))
    for patch, color in zip(box_plot.artists, colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Tùy chỉnh median line - làm nổi bật hơn
    for line in box_plot.lines:
        if len(line.get_xdata()) == 2:  # Median line
            # Đổi sang màu đỏ đậm, tăng độ dày, thêm outline đen
            line.set_color('#FF0000')  # Màu đỏ đậm
            line.set_linewidth(4.0)  # Tăng độ dày
            line.set_alpha(1.0)  # Độ trong suốt 100%
            # Thêm shadow effect (outline đen) để nổi bật
            line.set_path_effects([
                path_effects.withStroke(linewidth=6, foreground='black', alpha=0.5)
            ])
        elif len(line.get_xdata()) == 4:  # Whiskers
            line.set_color('black')
            line.set_linewidth(1.5)
    
    ax.set_xlabel('Genre', fontsize=13, fontweight='bold', color='#333')
    ax.set_ylabel('Average Rating', fontsize=13, fontweight='bold', color='#333')
    ax.set_title('Rating Distribution by Genre (Top 10)', fontsize=16, fontweight='bold', pad=20, color='#333')
    
    # Tùy chỉnh grid và background
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_ylim([0, 5.2])
    
    # Xoay labels
    plt.xticks(rotation=45, ha='right', fontsize=11, fontweight='bold')
    plt.yticks(fontsize=11)
    
    # Thêm giá trị median trên mỗi box
    medians = genre_rating_top.groupby('genre')['avg_rating'].median()
    for i, (genre, median) in enumerate(medians.items()):
        ax.text(i, median, f'{median:.2f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    plt.suptitle('')  # Bỏ title mặc định
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '12_rating_by_genre_boxplot.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("   ✅ Saved: 12_rating_by_genre_boxplot.png")
    return '12_rating_by_genre_boxplot.png'

# ============================================================================
# TẠO DASHBOARD HTML
# ============================================================================

def create_dashboard_html(plot_files, movies, ratings):
    """Tạo dashboard HTML tổng hợp tất cả biểu đồ với tabs"""
    print("\n" + "=" * 80)
    print("CREATING DASHBOARD HTML...")
    print("=" * 80)
    
    # Tính toán thống kê cho story telling
    avg_rating = ratings['rating'].mean()
    total_movies = len(movies)
    total_ratings = len(ratings)
    total_users = ratings['userId'].nunique()
    year_min = int(movies['year'].min()) if movies['year'].notna().any() else 1900
    year_max = int(movies['year'].max()) if movies['year'].notna().any() else 2020
    top_genre = movies['genres_list'].apply(
        lambda x: eval(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
    ).explode().value_counts().index[0]
    
    html_content = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MovieLens Dataset - Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 20px;
            color: #333;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            padding: 40px 30px;
            border-bottom: 3px solid #fff;
        }}
        
        .header h1 {{
            font-size: 2.8em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header p {{
            font-size: 1.2em;
            opacity: 0.95;
        }}
        
        .main-layout {{
            display: flex;
            min-height: calc(100vh - 200px);
        }}
        
        .sidebar {{
            width: 280px;
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 0;
            overflow-y: auto;
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
        }}
        
        .sidebar h3 {{
            padding: 15px 20px;
            font-size: 1.2em;
            border-bottom: 2px solid rgba(255,255,255,0.2);
            margin-bottom: 10px;
        }}
        
        .nav-menu {{
            list-style: none;
            padding: 0;
        }}
        
        .nav-item {{
            padding: 0;
        }}
        
        .nav-link {{
            display: block;
            padding: 12px 20px;
            color: white;
            text-decoration: none;
            transition: all 0.3s;
            border-left: 3px solid transparent;
            font-size: 0.95em;
        }}
        
        .nav-link:hover {{
            background: rgba(255,255,255,0.1);
            border-left: 3px solid white;
            padding-left: 25px;
        }}
        
        .nav-link.active {{
            background: rgba(255,255,255,0.2);
            border-left: 3px solid white;
            font-weight: bold;
        }}
        
        .content-area {{
            flex: 1;
            padding: 30px;
            overflow-y: auto;
        }}
        
        .plot-section {{
            scroll-margin-top: 20px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-card h3 {{
            font-size: 2.5em;
            margin-bottom: 8px;
        }}
        
        .stat-card p {{
            font-size: 1em;
            opacity: 0.95;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 3px solid #667eea;
        }}
        
        .plots-container {{
            display: flex;
            flex-direction: column;
            gap: 40px;
        }}
        
        .plot-card {{
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
            border-left: 5px solid #667eea;
        }}
        
        .plot-card:hover {{
            transform: translateX(5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }}
        
        .plot-card h3 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.4em;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .plot-card img {{
            width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }}
        
        .plot-card iframe {{
            width: 100%;
            height: 600px;
            border: none;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }}
        
        .interactive-badge {{
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.75em;
            margin-left: 10px;
            font-weight: bold;
        }}
        
        .story-section {{
            margin-bottom: 40px;
            padding: 25px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px;
            border-left: 5px solid #667eea;
        }}
        
        .story-section h3 {{
            color: #667eea;
            font-size: 1.5em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .story-section p {{
            line-height: 1.8;
            color: #555;
            font-size: 1.05em;
            text-align: justify;
        }}
        
        .story-highlight {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
        }}
        
        .story-highlight strong {{
            color: #856404;
        }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            color: #666;
            border-top: 2px solid #eee;
            margin-top: 40px;
            background: #f8f9fa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 MovieLens Dataset Dashboard</h1>
            <p>Data analysis and visualization of {total_movies:,} movies and {total_ratings:,} ratings</p>
        </div>
        
        <div class="main-layout">
            <!-- Sidebar Navigation -->
            <div class="sidebar">
                <h3>📊 Charts</h3>
                <ul class="nav-menu">
                    <li class="nav-item"><a href="#plot1" class="nav-link" onclick="scrollToSection('plot1')">⭐ Rating Distribution</a></li>
                    <li class="nav-item"><a href="#plot2" class="nav-link" onclick="scrollToSection('plot2')">🎭 Top Genres</a></li>
                    <li class="nav-item"><a href="#plot3" class="nav-link" onclick="scrollToSection('plot3')">🏆 Top Rated Movies</a></li>
                    <li class="nav-item"><a href="#plot4" class="nav-link" onclick="scrollToSection('plot4')">🌟 Highest Rated</a></li>
                    <li class="nav-item"><a href="#plot5" class="nav-link" onclick="scrollToSection('plot5')">📅 Ratings by Year</a></li>
                    <li class="nav-item"><a href="#plot6" class="nav-link" onclick="scrollToSection('plot6')">👥 User Distribution</a></li>
                    <li class="nav-item"><a href="#plot7" class="nav-link" onclick="scrollToSection('plot7')">🎬 Movie Distribution</a></li>
                    <li class="nav-item"><a href="#plot8" class="nav-link" onclick="scrollToSection('plot8')">🔥 Genre Heatmap</a></li>
                    <li class="nav-item"><a href="#plot9" class="nav-link" onclick="scrollToSection('plot9')">📊 Scatter Plot</a></li>
                    <li class="nav-item"><a href="#plot10" class="nav-link" onclick="scrollToSection('plot10')">📈 Ratings by Month</a></li>
                    <li class="nav-item"><a href="#plot11" class="nav-link" onclick="scrollToSection('plot11')">🎪 Genres by Year</a></li>
                    <li class="nav-item"><a href="#plot12" class="nav-link" onclick="scrollToSection('plot12')">📊 Rating by Genre</a></li>
                </ul>
            </div>
            
            <!-- Main Content -->
            <div class="content-area">
                <!-- Stats Overview -->
                <div id="overview" class="plot-section">
                    <div class="stats-grid">
                        <div class="stat-card">
                            <h3>🎬 {total_movies:,}</h3>
                            <p>Movies</p>
                        </div>
                        <div class="stat-card">
                            <h3>⭐ {total_ratings:,}</h3>
                            <p>Ratings</p>
                        </div>
                        <div class="stat-card">
                            <h3>👥 {total_users:,}</h3>
                            <p>Users</p>
                        </div>
                        <div class="stat-card">
                            <h3>🎭 19</h3>
                            <p>Genres</p>
                        </div>
                    </div>
                </div>
                
                <!-- Plots Section -->
                <div class="section">
                    <h2 class="section-title">📊 Data Analysis Charts</h2>
                    <div class="plots-container">
"""
    
    # Thêm các biểu đồ static
    plot_descriptions = {
        '1_rating_distribution.png': 'Rating Distribution',
        '2_top_genres.png': 'Top 15 Popular Genres',
        '3_top_rated_movies.png': 'Top 20 Most Rated Movies',
        '4_highest_rated_movies.png': 'Top 20 Highest Rated Movies',
        '5_ratings_by_year.png': 'Ratings by Year',
        '6_user_rating_distribution.png': 'User Rating Distribution',
        '7_movie_rating_distribution.png': 'Movie Rating Distribution',
        '8_genre_correlation_heatmap.png': 'Genre Correlation Heatmap',
        '12_rating_by_genre_boxplot.png': 'Rating Distribution by Genre'
    }
    
    # Icons cho từng biểu đồ
    plot_icons = {
        '1_rating_distribution.png': '⭐',
        '2_top_genres.png': '🎭',
        '3_top_rated_movies.png': '🏆',
        '4_highest_rated_movies.png': '🌟',
        '5_ratings_by_year.png': '📅',
        '6_user_rating_distribution.png': '👥',
        '7_movie_rating_distribution.png': '🎬',
        '8_genre_correlation_heatmap.png': '🔥',
        '12_rating_by_genre_boxplot.png': '📊'
    }
    
    # Mapping plot files to IDs
    plot_ids = {
        '1_rating_distribution.png': 'plot1',
        '2_top_genres.png': 'plot2',
        '3_top_rated_movies.png': 'plot3',
        '4_highest_rated_movies.png': 'plot4',
        '5_ratings_by_year.png': 'plot5',
        '6_user_rating_distribution.png': 'plot6',
        '7_movie_rating_distribution.png': 'plot7',
        '8_genre_correlation_heatmap.png': 'plot8',
        '12_rating_by_genre_boxplot.png': 'plot12'
    }
    
    for plot_file in plot_files:
        if plot_file.endswith('.png'):
            plot_name = plot_file.replace('.png', '')
            description = plot_descriptions.get(plot_file, plot_name.replace('_', ' ').title())
            icon = plot_icons.get(plot_file, '📈')
            plot_id = plot_ids.get(plot_file, '')
            html_content += f"""
                        <div id="{plot_id}" class="plot-section plot-card">
                            <h3>{icon} {description}</h3>
                            <img src="../{OUTPUT_DIR}/{plot_file}" alt="{description}">
                        </div>
"""
    
    html_content += """
                    </div>
                </div>
                
                <div class="section">
                    <h2 class="section-title">🎯 Interactive Charts <span class="interactive-badge">INTERACTIVE</span></h2>
                    <div class="plots-container">
"""
    
    # Icons cho biểu đồ interactive
    interactive_icons = {
        '9_scatter_rating_vs_count.html': '📊',
        '10_ratings_by_month.html': '📈',
        '11_top_genres_by_year.html': '🎪'
    }
    
    interactive_descriptions = {
        '9_scatter_rating_vs_count.html': 'Scatter Plot: Rating vs Number of Ratings',
        '10_ratings_by_month.html': 'Ratings by Month',
        '11_top_genres_by_year.html': 'Top Genres by Year'
    }
    
    interactive_ids = {
        '9_scatter_rating_vs_count.html': 'plot9',
        '10_ratings_by_month.html': 'plot10',
        '11_top_genres_by_year.html': 'plot11'
    }
    
    for plot_file in plot_files:
        if plot_file.endswith('.html'):
            description = interactive_descriptions.get(plot_file, plot_file.replace('_', ' ').title())
            icon = interactive_icons.get(plot_file, '📊')
            plot_id = interactive_ids.get(plot_file, '')
            html_content += f"""
                        <div id="{plot_id}" class="plot-section plot-card">
                            <h3>{icon} {description} <span class="interactive-badge">INTERACTIVE</span></h3>
                            <iframe src="../{OUTPUT_DIR}/{plot_file}"></iframe>
                        </div>
"""
    
    html_content += """
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            <p>🎬 MovieLens Dataset Analysis - Recommendation System Project</p>
        </div>
    </div>
    
    <script>
        function scrollToSection(sectionId) {{
            const section = document.getElementById(sectionId);
            if (section) {{
                // Update active nav link
                document.querySelectorAll('.nav-link').forEach(link => {{
                    link.classList.remove('active');
                }});
                event.target.classList.add('active');
                
                // Smooth scroll
                section.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
            }}
        }}
        
        // Highlight active section on scroll
        window.addEventListener('scroll', function() {{
            const sections = document.querySelectorAll('.plot-section');
            const navLinks = document.querySelectorAll('.nav-link');
            
            let current = '';
            sections.forEach(section => {{
                const sectionTop = section.offsetTop;
                const sectionHeight = section.clientHeight;
                if (window.pageYOffset >= (sectionTop - 200)) {{
                    current = section.getAttribute('id');
                }}
            }});
            
            navLinks.forEach(link => {{
                link.classList.remove('active');
                if (link.getAttribute('href') === '#' + current) {{
                    link.classList.add('active');
                }}
            }});
        }});
    </script>
</body>
</html>
"""
    
    # Lưu file
    dashboard_path = os.path.join(DASHBOARD_DIR, 'dashboard.html')
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"   ✅ Dashboard saved: {dashboard_path}")
    return dashboard_path

def main():
    """Hàm chính"""
    print("\n" + "=" * 80)
    print("VISUALIZATION - MOVIELENS DATASET")
    print("=" * 80)
    print("\nTạo 12 biểu đồ (9 static + 3 interactive)")
    print("Tổng hợp vào dashboard HTML")
    print("=" * 80)
    
    # Load data
    movies, ratings, tags = load_data()
    
    # Tạo các biểu đồ
    plot_files = []
    
    # Static plots
    plot_files.append(plot1_rating_distribution(ratings))
    plot_files.append(plot2_top_genres(movies))
    plot_files.append(plot3_top_rated_movies(movies))
    plot_files.append(plot4_highest_rated_movies(movies))
    plot_files.append(plot5_ratings_by_year(ratings))
    plot_files.append(plot6_user_rating_distribution(ratings))
    plot_files.append(plot7_movie_rating_distribution(movies))
    plot_files.append(plot8_genre_correlation_heatmap(movies))
    plot_files.append(plot12_rating_by_genre_boxplot(movies))
    
    # Interactive plots
    plot_files.append(plot9_scatter_rating_vs_count(movies))
    plot_files.append(plot10_ratings_by_month(ratings))
    plot_files.append(plot11_top_genres_by_year(movies, ratings))
    
    # Tạo dashboard
    dashboard_path = create_dashboard_html(plot_files, movies, ratings)
    
    # Tóm tắt
    print("\n" + "=" * 80)
    print("TÓM TẮT")
    print("=" * 80)
    print(f"\n✅ Đã tạo {len(plot_files)} biểu đồ:")
    print(f"   - 9 biểu đồ static (PNG)")
    print(f"   - 3 biểu đồ interactive (HTML)")
    print(f"\n📁 Files:")
    print(f"   - Images: {OUTPUT_DIR}/")
    print(f"   - Dashboard: {dashboard_path}")
    print(f"\n🌐 Mở dashboard:")
    print(f"   - File: {dashboard_path}")
    print(f"   - Hoặc: file:///{os.path.abspath(dashboard_path)}")
    
    print("\n" + "=" * 80)
    print("HOÀN THÀNH!")
    print("=" * 80)

if __name__ == "__main__":
    main()

