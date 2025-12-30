"""
Streamlit App - Movie Recommendation System
Cơ bản với chỗ để implement real-time recommendations sau
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import requests
import os
import json

# Import từ package
from recommender.models import MovieSearchEngine, HybridRecommender, ContentBasedRecommender

# Page config
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Theme với Movie Style
st.markdown("""
    <style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Dark theme background - Nền đen */
    .stApp {
        background: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Main content area */
    .main .block-container {
        background: #000000;
        color: #ffffff;
    }
    
    /* Text color - Chữ trắng */
    .stMarkdown, p, div, span, h1, h2, h3, h4, h5, h6, label {
        color: #ffffff !important;
    }
    
    /* Streamlit text elements */
    .stText, .stMarkdown p, .stMarkdown div {
        color: #ffffff !important;
    }
    
    /* Info/Success/Warning messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        color: #ffffff !important;
    }
    
    /* Movie card styling */
    .movie-card {
        background: linear-gradient(135deg, #2a2a3e 0%, #1a1a2e 100%);
        padding: 1rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 215, 0, 0.5);
    }
    
    .movie-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #ffd700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    .movie-info {
        color: #b8b8b8;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Section headers */
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #ffd700;
        margin: 2rem 0 1rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    /* Buttons - Đỏ Netflix */
    .stButton>button {
        background: #E50914 !important;
        color: white !important;
        border: none !important;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .stButton>button:hover {
        background: #F40612 !important;
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(229, 9, 20, 0.4);
    }
    
    /* Button primary (active) */
    button[kind="primary"] {
        background: #E50914 !important;
        color: white !important;
    }
    
    button[kind="secondary"] {
        background: rgba(229, 9, 20, 0.3) !important;
        color: white !important;
        border: 1px solid #E50914 !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #2a2a3e 100%);
    }
    
    /* Text inputs */
    .stTextInput>div>div>input {
        background-color: #2a2a3e;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
    }
    
    /* Slider */
    .stSlider>div>div>div {
        background: linear-gradient(90deg, #ffd700 0%, #ffed4e 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2a2a3e;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #b8b8b8;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ffd700;
    }
    
    /* Success/Info messages */
    .stSuccess {
        background-color: rgba(46, 125, 50, 0.2);
        border-left: 4px solid #4caf50;
    }
    
    .stInfo {
        background-color: rgba(33, 150, 243, 0.2);
        border-left: 4px solid #2196f3;
    }
    
    /* Rating stars */
    .rating-star {
        color: #ffd700;
        font-size: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_history' not in st.session_state:
    st.session_state.user_history = {}  # {movieId: rating}

if 'search_engine' not in st.session_state:
    st.session_state.search_engine = None

if 'hybrid_recommender' not in st.session_state:
    st.session_state.hybrid_recommender = None

if 'content_recommender' not in st.session_state:
    st.session_state.content_recommender = None

if 'tmdb_mapping' not in st.session_state:
    st.session_state.tmdb_mapping = None

@st.cache_data
def load_tmdb_mapping():
    """Load TMDB ID mapping"""
    links = pd.read_csv("data_cleaned/links_cleaned.csv")
    # Tạo mapping movieId -> tmdbId
    mapping = dict(zip(links['movieId'], links['tmdbId']))
    return mapping

@st.cache_data
def load_posters_from_file():
    """Load posters từ file JSON - CHỈ LOCAL PATHS"""
    poster_file = "data_cleaned/posters_mapping.json"
    if os.path.exists(poster_file):
        try:
            with open(poster_file, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            
            # CHỈ LẤY LOCAL PATHS, LOẠI BỎ URL
            local_paths = {}
            for movie_id, path_or_url in mapping.items():
                # Bỏ qua URL
                if isinstance(path_or_url, str) and path_or_url.startswith('http'):
                    continue
                # Chỉ lấy local path và file tồn tại
                if isinstance(path_or_url, str) and not path_or_url.startswith('http') and os.path.exists(path_or_url) and os.path.getsize(path_or_url) > 0:
                    local_paths[movie_id] = path_or_url
            
            return local_paths
        except Exception as e:
            print(f"⚠️ Error loading posters mapping: {e}")
            return {}
    return {}

# Load posters sau khi hàm được định nghĩa
if 'poster_cache' not in st.session_state or len(st.session_state.poster_cache) == 0:
    st.session_state.poster_cache = load_posters_from_file()

@st.cache_data
def get_search_suggestions(query, limit=10):
    """Lấy suggestions cho autocomplete khi gõ vào search box"""
    if not query or len(query.strip()) < 2:
        return []
    
    if st.session_state.search_engine is None:
        return []
    
    # Dùng search_exact để nhanh hơn (contains search)
    results = st.session_state.search_engine.search_exact(query, limit=limit)
    
    if len(results) == 0:
        return []
    
    # Trả về list titles
    suggestions = results['title_clean'].tolist()
    return suggestions

@st.cache_resource
def load_models():
    """Load models (cached để không load lại mỗi lần)"""
    with st.spinner("Loading models... This may take a few seconds..."):
        search_engine = MovieSearchEngine()
        hybrid_recommender = HybridRecommender()
        content_recommender = ContentBasedRecommender(use_hybrid=True)
    return search_engine, hybrid_recommender, content_recommender

def display_movie_card_compact(movie, show_rating=True, prefix=""):
    """Hiển thị movie card compact (cho grid layout)"""
    try:
        movie_id = movie['movieId']
        
        # Lấy poster - CHỈ TỪ LOCAL
        poster_path = st.session_state.poster_cache.get(str(movie_id))
        if poster_path and isinstance(poster_path, str) and not poster_path.startswith('http') and os.path.exists(poster_path):
            poster_url = poster_path
        else:
            poster_url = None
        # Không gọi API - chỉ dùng local hoặc placeholder
        
        # Compact card với poster
        with st.container():
            try:
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    # Placeholder với kích thước bằng poster (200x300)
                    st.image("https://via.placeholder.com/200x300/2a2a3e/ffd700?text=No+Poster", use_container_width=True)
            except:
                st.image("https://via.placeholder.com/200x300/2a2a3e/ffd700?text=No+Poster", use_container_width=True)
            
            # Title
            title = movie.get('title_clean', movie.get('title', 'Unknown'))[:30] + "..." if len(movie.get('title_clean', movie.get('title', ''))) > 30 else movie.get('title_clean', movie.get('title', 'Unknown'))
            st.markdown(f"<div style='color: #ffd700; font-weight: bold; font-size: 0.9rem; margin-top: 0.5rem;'>{title}</div>", unsafe_allow_html=True)
            
            # Year & Genres
            year = int(movie.get('year', 0)) if pd.notna(movie.get('year')) else 0
            genres_display = movie.get('genres', 'Unknown')[:20] + "..." if len(movie.get('genres', '')) > 20 else movie.get('genres', 'Unknown')
            info = f"<div style='color: #b8b8b8; font-size: 0.8rem;'>{year} • {genres_display}</div>"
            if show_rating and 'avg_rating' in movie and pd.notna(movie['avg_rating']):
                info += f"<div style='color: #ffd700; font-size: 0.8rem; margin-top: 0.2rem;'>⭐ {movie['avg_rating']:.1f}</div>"
            st.markdown(info, unsafe_allow_html=True)
            
            # View button
            unique_key = f"{prefix}_view_{movie_id}" if prefix else f"view_{movie_id}"
            if st.button("View", key=unique_key, use_container_width=True):
                st.session_state.selected_movie_id = movie_id
                # Đánh dấu để scroll lên đầu khi chuyển phim
                st.session_state.scroll_to_top = True
                st.rerun()
    except Exception as e:
        st.error(f"Error displaying movie card: {str(e)}")

def display_movie_card(movie, show_rating=True, prefix=""):
    """Hiển thị movie card full (cho detail pages)"""
    movie_id = movie['movieId']
    
    # Lấy poster - CHỈ TỪ LOCAL
    poster_path = st.session_state.poster_cache.get(str(movie_id))
    if poster_path and isinstance(poster_path, str) and not poster_path.startswith('http') and os.path.exists(poster_path):
        poster_url = poster_path
    else:
        poster_url = None
    # Không gọi API - chỉ dùng local hoặc placeholder
    
    # Full card layout
    if poster_url:
        col_poster, col_info = st.columns([1, 3])
        
        with col_poster:
            st.image(poster_url, width=150, use_container_width=False)
        
        with col_info:
            st.markdown(f"<div class='movie-title'>{movie['title_clean']}</div>", unsafe_allow_html=True)
            info = f"Year: {int(movie['year'])} | Genres: {movie['genres']}"
            if show_rating and 'avg_rating' in movie and pd.notna(movie['avg_rating']):
                info += f" | ⭐ {movie['avg_rating']:.1f}"
            st.markdown(f"<div class='movie-info'>{info}</div>", unsafe_allow_html=True)
            
            if show_rating:
                unique_key = f"{prefix}_view_{movie_id}" if prefix else f"view_{movie_id}"
                if st.button("View", key=unique_key):
                    st.session_state.selected_movie_id = movie_id
                    st.rerun()
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"<div class='movie-title'>{movie['title_clean']}</div>", unsafe_allow_html=True)
            info = f"Year: {int(movie['year'])} | Genres: {movie['genres']}"
            if show_rating and 'avg_rating' in movie and pd.notna(movie['avg_rating']):
                info += f" | ⭐ {movie['avg_rating']:.1f}"
            st.markdown(f"<div class='movie-info'>{info}</div>", unsafe_allow_html=True)
        
        with col2:
            if show_rating:
                unique_key = f"{prefix}_view_{movie_id}" if prefix else f"view_{movie_id}"
                if st.button("View", key=unique_key):
                    st.session_state.selected_movie_id = movie_id
                    st.rerun()

def home_page():
    """Home page - Popular movies với UI đẹp"""
    # Pre-load posters
    if st.session_state.tmdb_mapping is None:
        st.session_state.tmdb_mapping = load_tmdb_mapping()
    
    if len(st.session_state.poster_cache) == 0:
        st.info("Chưa có posters trong file. Chạy `python download_posters_images_v2.py` để tải posters!")
    
    # Popular Movies Section
    st.markdown("""
        <div class='section-header'>Popular Picks</div>
    """, unsafe_allow_html=True)
    
    popular = st.session_state.search_engine.get_popular_movies(limit=20)
    
    cols = st.columns(5)
    for idx, (i, movie) in enumerate(popular.iterrows()):
        if idx >= 20:
            break
        with cols[idx % 5]:
            display_movie_card_compact(movie, prefix="popular")

def search_page():
    """Search page với real-time autocomplete và fuzzy matching"""
    # Header
    st.markdown("""
        <div style='font-size: 2rem; font-weight: bold; color: #ffd700; 
                   text-shadow: 2px 2px 4px rgba(0,0,0,0.5); margin-bottom: 1rem;'>
            Search Movies
        </div>
    """, unsafe_allow_html=True)
    
    # Search box với real-time autocomplete
    query = st.text_input("", placeholder="Enter movie title... (e.g., Avatar, Toy Story)", key="search_input")
    
    # Real-time suggestions khi đang gõ (hiển thị ngay khi gõ)
    selected_suggestion = None
    suggestions = []
    
    if query and len(query) >= 2:
        # Lấy suggestions real-time
        suggestions = get_search_suggestions(query, limit=10)
        
        if len(suggestions) > 0:
            # Tạo options cho selectbox
            options = ["-- Select a movie --"] + suggestions
            
            # Selectbox để chọn suggestion ngay
            selected_idx = st.selectbox(
                "Suggestions (select to search immediately):",
                range(len(options)),
                format_func=lambda x: options[x] if x < len(options) else "",
                key="search_suggestion_select"
            )
            
            # Nếu chọn suggestion (không phải "-- Select a movie --")
            if selected_idx > 0:
                selected_suggestion = options[selected_idx]
                # Tự động set search query
                st.session_state.search_query = selected_suggestion
                query = selected_suggestion  # Update query để search ngay
    
    # Lấy query để search (ưu tiên selected_suggestion, sau đó là query input)
    if selected_suggestion:
        current_query = selected_suggestion
    elif query and len(query) >= 2:
        current_query = query
        # Tự động update search_query khi gõ
        if 'search_query' not in st.session_state or st.session_state.search_query != query:
            st.session_state.search_query = query
    elif 'search_query' in st.session_state and st.session_state.search_query:
        current_query = st.session_state.search_query
    else:
        current_query = ""
    
    # Thực hiện search với fuzzy matching
    results = pd.DataFrame()
    if current_query and len(current_query) >= 2:
        try:
            with st.spinner("Searching..."):
                # Dùng search() với fuzzy matching để tìm phim giống hoặc na ná
                results = st.session_state.search_engine.search(current_query, limit=20)
            
            if len(results) > 0:
                st.success(f"✅ Found {len(results)} results for '{current_query}'")
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Display results in grid
                cols = st.columns(5)
                for idx, (i, movie) in enumerate(results.iterrows()):
                    if idx >= 20:
                        break
                    with cols[idx % 5]:
                        display_movie_card_compact(movie, prefix="search")
            else:
                st.warning("⚠️ No results found. Try a different search term.")
        except Exception as e:
            st.error(f"❌ Error searching: {str(e)}")
            results = pd.DataFrame()
    
    # Related movies nếu có kết quả - DÙNG CONTENT-BASED
    if current_query and len(results) > 0:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
            <div class='section-header'>Related Movies</div>
        """, unsafe_allow_html=True)
        
        # Lấy similar movies từ phim đầu tiên
        try:
            first_movie_id = results.iloc[0]['movieId']
            with st.spinner("Finding similar movies..."):
                similar = st.session_state.content_recommender.get_similar_movies(first_movie_id, n=10)
            
            if len(similar) > 0:
                cols = st.columns(5)
                for idx, (i, movie) in enumerate(similar.iterrows()):
                    if idx >= 10:
                        break
                    with cols[idx % 5]:
                        display_movie_card_compact(movie, prefix="similar", show_rating=False)
        except Exception as e:
            st.warning(f"⚠️ Could not load similar movies: {str(e)}")

def movie_detail_page(movie_id):
    """Movie detail page - movie_id có thể thay đổi khi click vào similar/recommended movies"""
    # Scroll lên đầu nếu vừa chuyển phim - SỬ DỤNG components.html
    if st.session_state.get("scroll_to_top", False):
        components.html(
            "<script>setTimeout(()=>window.parent.scrollTo({top:0,left:0,behavior:'smooth'}),50);setTimeout(()=>window.parent.scrollTo({top:0,left:0,behavior:'smooth'}),250);</script>",
            height=0,
            width=0
        )
        st.session_state.scroll_to_top = False
    
    movie = st.session_state.search_engine.movies[
        st.session_state.search_engine.movies['movieId'] == movie_id
    ]
    
    if len(movie) == 0:
        st.error("Movie not found!")
        return
    
    movie = movie.iloc[0]
    
    # Load TMDB mapping
    if st.session_state.tmdb_mapping is None:
        st.session_state.tmdb_mapping = load_tmdb_mapping()
    
    # Lấy poster - CHỈ TỪ LOCAL
    poster_path = st.session_state.poster_cache.get(str(movie_id))
    if poster_path and isinstance(poster_path, str) and not poster_path.startswith('http') and os.path.exists(poster_path):
        poster_url = poster_path
    else:
        poster_url = None
    # Không gọi API - chỉ dùng local hoặc placeholder
    
    # Header với poster
    col_poster, col_info = st.columns([2, 3])
    
    with col_poster:
        if poster_url:
            st.image(poster_url, width=300, use_container_width=False)
        else:
            # Placeholder với kích thước bằng poster (300x450)
            st.image("https://via.placeholder.com/300x450/2a2a3e/ffd700?text=No+Poster", width=300)
    
    with col_info:
        st.markdown(f"""
            <div style='font-size: 2.5rem; font-weight: bold; color: #ffd700; 
                       text-shadow: 3px 3px 6px rgba(0,0,0,0.5); margin-bottom: 1rem;'>
                {movie['title_clean']}
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<div style='color: #b8b8b8; font-size: 1.1rem; margin-bottom: 0.5rem;'><strong>Year:</strong> {int(movie['year'])}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: #b8b8b8; font-size: 1.1rem; margin-bottom: 0.5rem;'><strong>Genres:</strong> {movie['genres']}</div>", unsafe_allow_html=True)
        if pd.notna(movie['avg_rating']):
            st.markdown(f"<div style='color: #ffd700; font-size: 1.1rem; margin-bottom: 1.5rem;'><strong>Average Rating:</strong> ⭐ {movie['avg_rating']:.2f} ({movie['num_ratings']:.0f} ratings)</div>", unsafe_allow_html=True)
        
        # Rating widget - Star buttons (ngay dưới average rating)
        st.markdown("""
            <div style='color: #ffd700; font-weight: bold; font-size: 1.2rem; margin-bottom: 0.5rem;'>
                Rate this movie:
            </div>
        """, unsafe_allow_html=True)
        
        # Lấy rating hiện tại
        current_rating = st.session_state.user_history.get(movie_id, 0.0)
        
        # Star buttons (5 sao)
        star_cols = st.columns(5)
        selected_rating = current_rating
        
        for i in range(1, 6):
            with star_cols[i-1]:
                rating_value = float(i)
                # Hiển thị sao đầy nếu đã rate >= rating_value, sao rỗng nếu chưa
                if current_rating >= rating_value:
                    star_display = "⭐"
                else:
                    star_display = "☆"
                
                # Button để chọn rating
                if st.button(star_display, key=f"star_{movie_id}_{i}", use_container_width=True):
                    selected_rating = rating_value
                    st.session_state.user_history[movie_id] = rating_value
                    st.success(f"✅ Rated {rating_value} stars!")
                    st.info("Recommendations will update in real-time!")
                    st.rerun()
        
        # Hiển thị rating đã chọn
        if current_rating > 0:
            stars_display = "⭐" * int(current_rating) + "☆" * (5 - int(current_rating))
            st.markdown(f"<div style='color: #ffd700; font-size: 1.2rem; margin-top: 0.5rem;'>{stars_display} <span style='color: #b8b8b8; font-size: 1rem;'>({current_rating} / 5.0)</span></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Recommended for you (Hybrid) - Dựa trên user_history (sẽ UPDATE khi rate) - LÊN TRƯỚC
    st.markdown("""
        <div class='section-header'>Recommended for You</div>
    """, unsafe_allow_html=True)
    
    # Logic: Lần đầu rate (1 rating) → KHÔNG hiển thị
    # Lần 2 rate (2 ratings) → HIỂN THỊ lần đầu
    # Lần 3+ rate → CẬP NHẬT
    if len(st.session_state.user_history) < 2:
        if len(st.session_state.user_history) == 0:
            st.info("Rate some movies to get personalized recommendations!")
        else:
            st.info("Rate one more movie to get personalized recommendations!")
    else:
        try:
            with st.spinner("Getting recommendations..."):
                # Dựa trên user_history - sẽ UPDATE khi rate phim mới
                recommendations = st.session_state.hybrid_recommender.recommend(
                    st.session_state.user_history,
                    n=10
                )
            
            if len(recommendations) > 0:
                st.info("These recommendations update in real-time as you rate more movies")
                st.markdown("<br>", unsafe_allow_html=True)
                cols = st.columns(5)
                for idx, (i, rec_movie) in enumerate(recommendations.iterrows()):
                    if idx >= 10:
                        break
                    with cols[idx % 5]:
                        # Click vào phim recommended → chuyển sang detail page của phim đó
                        # Recommended sẽ UPDATE khi rate (dựa trên user_history mới)
                        display_movie_card_compact(rec_movie, show_rating=False, prefix="detail_rec")
            else:
                st.info("No recommendations available.")
        except Exception as e:
            st.warning(f"⚠️ Could not load recommendations: {str(e)}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Similar movies (Content-based) - LUÔN dựa trên phim hiện tại đang xem - XUỐNG SAU
    st.markdown("""
        <div class='section-header'>Similar Movies</div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div style='color: #b8b8b8; font-size: 0.9rem; margin-bottom: 1rem;'>
            Movies similar to <strong style='color: #ffd700;'>{movie['title_clean']}</strong>
        </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Finding similar movies..."):
        try:
            # LUÔN dùng movie_id hiện tại (phim đang xem)
            similar = st.session_state.content_recommender.get_similar_movies(movie_id, n=10)
            
            if len(similar) > 0:
                cols = st.columns(5)
                for idx, (i, sim_movie) in enumerate(similar.iterrows()):
                    if idx >= 10:
                        break
                    with cols[idx % 5]:
                        # Click vào phim similar → chuyển sang detail page của phim đó
                        display_movie_card_compact(sim_movie, show_rating=False, prefix="similar")
            else:
                st.info("No similar movies found.")
        except Exception as e:
            st.warning(f"⚠️ Could not load similar movies: {str(e)}")

def recommendations_page():
    """My Recommendations page - Grid layout như home page"""
    st.markdown("""
        <div style='font-size: 2rem; font-weight: bold; color: #ffd700; 
                   text-shadow: 2px 2px 4px rgba(0,0,0,0.5); margin-bottom: 1rem;'>
            My Recommendations
        </div>
    """, unsafe_allow_html=True)
    
    if len(st.session_state.user_history) == 0:
        st.info("Rate some movies to get personalized recommendations!")
        st.markdown("""
        **How it works:**
        1. Go to Home or Search
        2. Click "View" on a movie
        3. Rate the movie
        4. Get personalized recommendations!
        """)
    else:
        st.markdown(f"""
            <div style='color: #b8b8b8; font-size: 1.1rem; margin-bottom: 1.5rem;'>
                Based on <strong style='color: #ffd700;'>{len(st.session_state.user_history)}</strong> movies you've rated
            </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Getting recommendations..."):
            recommendations = st.session_state.hybrid_recommender.recommend(
                st.session_state.user_history,
                n=20
            )
        
        if len(recommendations) > 0:
            st.info("Recommendations update in real-time as you rate more movies")
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Grid layout như home page (5 cột)
            cols = st.columns(5)
            for idx, (i, rec_movie) in enumerate(recommendations.iterrows()):
                if idx >= 20:
                    break
                with cols[idx % 5]:
                    display_movie_card_compact(rec_movie, prefix="recommendations")
        else:
            st.info("No recommendations available.")

def history_page():
    """My History page với UI đẹp"""
    st.markdown("""
        <div style='font-size: 2rem; font-weight: bold; color: #ffd700; 
                   text-shadow: 2px 2px 4px rgba(0,0,0,0.5); margin-bottom: 1rem;'>
            My Rating History
        </div>
    """, unsafe_allow_html=True)
    
    if len(st.session_state.user_history) == 0:
        st.info("You haven't rated any movies yet!")
    else:
        st.markdown(f"""
            <div style='color: #b8b8b8; font-size: 1.1rem; margin-bottom: 1.5rem;'>
                You've rated <strong style='color: #ffd700;'>{len(st.session_state.user_history)}</strong> movies
            </div>
        """, unsafe_allow_html=True)
        
        # Display history in grid
        cols = st.columns(5)
        history_list = list(st.session_state.user_history.items())
        for idx, (movie_id, rating) in enumerate(history_list):
            movie = st.session_state.search_engine.movies[
                st.session_state.search_engine.movies['movieId'] == movie_id
            ]
            if len(movie) > 0:
                movie = movie.iloc[0]
                with cols[idx % 5]:
                    display_movie_card_compact(movie, show_rating=False, prefix=f"history_{movie_id}")
                    st.markdown(f"""
                        <div style='text-align: center; color: #ffd700; font-weight: bold; margin-top: 0.5rem;'>
                            ⭐ {rating}
                        </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear History", use_container_width=True):
            st.session_state.user_history = {}
            st.success("✅ History cleared!")
            st.rerun()

def render_header():
    """Render header với navigation tabs - hiển thị ở mọi page"""
    # Anchor ở đầu trang để scroll đến
    st.markdown('<div id="page-top-anchor" style="position: absolute; top: 0;"></div>', unsafe_allow_html=True)
    
    # Initialize page nếu chưa có
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    
    # Kiểm tra xem có đang ở movie detail page không
    is_movie_detail = 'selected_movie_id' in st.session_state
    
    # Header với logo và navigation
    if is_movie_detail:
        # Nếu đang ở movie detail, hiển thị nút Back và logo
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("Back", use_container_width=True, type="primary"):
                del st.session_state.selected_movie_id
                st.rerun()
        
        with col2:
            st.markdown("""
                <div style='font-size: 2rem; font-weight: bold; color: #ffd700; 
                           text-shadow: 2px 2px 4px rgba(0,0,0,0.5); margin-bottom: 1rem;'>
                    MovieRec
                </div>
            """, unsafe_allow_html=True)
    else:
        # Nếu không phải movie detail, hiển thị logo và navigation tabs
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown("""
                <div style='font-size: 2rem; font-weight: bold; color: #ffd700; 
                           text-shadow: 2px 2px 4px rgba(0,0,0,0.5); margin-bottom: 1rem;'>
                    MovieRec
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            # Navigation buttons
            nav_cols = st.columns(4)
            
            with nav_cols[0]:
                if st.button("Home", use_container_width=True, type="primary" if st.session_state.current_page == "Home" else "secondary"):
                    st.session_state.current_page = "Home"
                    st.rerun()
            
            with nav_cols[1]:
                if st.button("Search", use_container_width=True, type="primary" if st.session_state.current_page == "Search" else "secondary"):
                    st.session_state.current_page = "Search"
                    st.rerun()
            
            with nav_cols[2]:
                if st.button("History", use_container_width=True, type="primary" if st.session_state.current_page == "History" else "secondary"):
                    st.session_state.current_page = "History"
                    st.rerun()
            
            with nav_cols[3]:
                if st.button("Recommend", use_container_width=True, type="primary" if st.session_state.current_page == "Recommend" else "secondary"):
                    st.session_state.current_page = "Recommend"
                    st.rerun()
    
    st.markdown("---")

def main():
    """Main app"""
    # Initialize models
    if st.session_state.search_engine is None:
        search_engine, hybrid_recommender, content_recommender = load_models()
        st.session_state.search_engine = search_engine
        st.session_state.hybrid_recommender = hybrid_recommender
        st.session_state.content_recommender = content_recommender
    
    # Render header LUÔN (ở mọi page, kể cả movie detail)
    render_header()
    
    # Check if movie detail page
    if 'selected_movie_id' in st.session_state:
        movie_detail_page(st.session_state.selected_movie_id)
        return
    
    # Get current page
    page = st.session_state.get('current_page', 'Home')
    
    # Clear search_query khi chuyển page (trừ khi đang ở Search page)
    if page != "Search" and 'search_query' in st.session_state:
        del st.session_state.search_query
    
    # Route to pages
    try:
        if page == "Home":
            home_page()
        elif page == "Search":
            search_page()
        elif page == "Recommend":
            recommendations_page()
        elif page == "History":
            history_page()
    except Exception as e:
        st.error(f"❌ Error loading page: {str(e)}")
        st.info("Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    main()

