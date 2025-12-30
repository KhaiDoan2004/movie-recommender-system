"""
Evaluation Module - Đánh giá Recommendation Models
Metrics: RMSE, MAE, Precision@K, Recall@K
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

# Đường dẫn - relative từ project root
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = str(BASE_DIR / "data_cleaned")

class RecommendationEvaluator:
    """Class để đánh giá recommendation models"""
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Khởi tạo evaluator
        
        Args:
            test_size: Tỷ lệ test set
            random_state: Random seed
        """
        print("=" * 80)
        print("INITIALIZING EVALUATOR")
        print("=" * 80)
        
        # Load ratings
        self.ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings_cleaned.csv"))
        print(f"✅ Loaded {len(self.ratings)} ratings")
        
        # Split train/test
        self.train_ratings, self.test_ratings = train_test_split(
            self.ratings,
            test_size=test_size,
            random_state=random_state,
            stratify=None
        )
        print(f"✅ Train set: {len(self.train_ratings)} ratings")
        print(f"✅ Test set: {len(self.test_ratings)} ratings")
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load recommendation models"""
        print("\n📦 Loading recommendation models...")
        
        from recommender.models import ContentBasedRecommender, CollaborativeRecommender, HybridRecommender
        
        print("   → Loading Content-Based...")
        self.content_recommender = ContentBasedRecommender(use_hybrid=True)
        
        print("   → Loading Collaborative...")
        # Cần train lại SVD trên train set (sẽ làm sau)
        self.collab_recommender = None  # Sẽ train lại
        
        print("   → Loading Hybrid...")
        self.hybrid_recommender = HybridRecommender()
        
        print("   ✅ Models loaded")
    
    def calculate_rmse_mae(self, y_true, y_pred):
        """
        Tính RMSE và MAE
        
        Args:
            y_true: True ratings
            y_pred: Predicted ratings
        
        Returns:
            rmse, mae
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return rmse, mae
    
    def evaluate_collaborative_rmse_mae(self):
        """
        Đánh giá Collaborative Filtering với RMSE và MAE
        Train SVD trên train set, test trên test set
        """
        print("\n" + "=" * 80)
        print("EVALUATING COLLABORATIVE FILTERING (RMSE, MAE)")
        print("=" * 80)
        
        from surprise import SVD, Dataset, Reader
        
        # Train SVD trên train set
        print("\n🔄 Training SVD on train set...")
        reader = Reader(rating_scale=(0.5, 5.0))
        train_data = Dataset.load_from_df(
            self.train_ratings[['userId', 'movieId', 'rating']],
            reader
        )
        trainset = train_data.build_full_trainset()
        
        svd = SVD(n_factors=50, random_state=42, verbose=False)
        svd.fit(trainset)
        print("   ✅ SVD trained")
        
        # Predict trên test set
        print("\n🔄 Predicting on test set...")
        predictions = []
        actuals = []
        
        for idx, row in self.test_ratings.iterrows():
            try:
                pred = svd.predict(row['userId'], row['movieId'])
                predictions.append(pred.est)
                actuals.append(row['rating'])
            except:
                # Nếu không predict được, dùng average rating
                predictions.append(self.train_ratings['rating'].mean())
                actuals.append(row['rating'])
        
        # Tính metrics
        rmse, mae = self.calculate_rmse_mae(actuals, predictions)
        
        print(f"\n📊 Results:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE:  {mae:.4f}")
        
        return {'rmse': rmse, 'mae': mae, 'predictions': predictions, 'actuals': actuals}
    
    def precision_at_k(self, recommended, relevant, k):
        """
        Tính Precision@K
        
        Args:
            recommended: List of recommended movie IDs
            relevant: Set of relevant movie IDs (user đã rate cao)
            k: Top K recommendations
        
        Returns:
            Precision@K
        """
        recommended_k = recommended[:k]
        if len(recommended_k) == 0:
            return 0.0
        
        relevant_recommended = len(set(recommended_k) & set(relevant))
        return relevant_recommended / len(recommended_k)
    
    def recall_at_k(self, recommended, relevant, k):
        """
        Tính Recall@K
        
        Args:
            recommended: List of recommended movie IDs
            relevant: Set of relevant movie IDs
            k: Top K recommendations
        
        Returns:
            Recall@K
        """
        recommended_k = recommended[:k]
        if len(relevant) == 0:
            return 0.0
        
        relevant_recommended = len(set(recommended_k) & set(relevant))
        return relevant_recommended / len(relevant)
    
    def evaluate_precision_recall_at_k(self, recommender, recommender_name, k=10, threshold=4.0, n_users=50):
        """
        Đánh giá Precision@K và Recall@K
        
        Args:
            recommender: Recommender object
            recommender_name: Tên recommender
            k: Top K recommendations
            threshold: Rating threshold để coi là relevant (>= threshold)
            n_users: Số users để test (để nhanh hơn)
        
        Returns:
            dict với precision@k và recall@k
        """
        print(f"\n" + "=" * 80)
        print(f"EVALUATING {recommender_name.upper()} (Precision@{k}, Recall@{k})")
        print("=" * 80)
        
        # Lấy sample users từ test set
        test_users = self.test_ratings['userId'].unique()[:n_users]
        print(f"\n📊 Testing on {len(test_users)} users...")
        
        precisions = []
        recalls = []
        
        for user_id in test_users:
            # Lấy ratings của user trong train set (để làm history)
            user_train_ratings = self.train_ratings[
                self.train_ratings['userId'] == user_id
            ]
            
            # Lấy ratings của user trong test set (để làm ground truth)
            user_test_ratings = self.test_ratings[
                self.test_ratings['userId'] == user_id
            ]
            
            # Relevant movies: phim user rate >= threshold trong test set
            relevant_movies = set(
                user_test_ratings[user_test_ratings['rating'] >= threshold]['movieId'].tolist()
            )
            
            if len(relevant_movies) == 0:
                continue  # Skip nếu không có relevant movies
            
            # Tạo user history từ train set
            user_history = dict(zip(
                user_train_ratings['movieId'],
                user_train_ratings['rating']
            ))
            
            # Get recommendations
            try:
                if recommender_name == "Content-Based":
                    recommendations_df = recommender.recommend_for_user_content_only(
                        user_history, n=k
                    )
                elif recommender_name == "Hybrid":
                    recommendations_df = recommender.recommend(user_history, n=k)
                elif recommender_name == "Collaborative":
                    recommendations_df = recommender.recommend_for_user(user_history, n=k)
                else:
                    recommendations_df = recommender.recommend_for_user(user_history, n=k)
                
                if len(recommendations_df) > 0:
                    recommended_movies = recommendations_df['movieId'].tolist()
                    
                    # Tính precision và recall
                    precision = self.precision_at_k(recommended_movies, relevant_movies, k)
                    recall = self.recall_at_k(recommended_movies, relevant_movies, k)
                    
                    precisions.append(precision)
                    recalls.append(recall)
            except Exception as e:
                # Skip nếu có lỗi
                continue
        
        # Tính average
        avg_precision = np.mean(precisions) if len(precisions) > 0 else 0.0
        avg_recall = np.mean(recalls) if len(recalls) > 0 else 0.0
        
        print(f"\n📊 Results:")
        print(f"   Precision@{k}: {avg_precision:.4f}")
        print(f"   Recall@{k}:    {avg_recall:.4f}")
        print(f"   Users evaluated: {len(precisions)}")
        
        return {
            'precision@k': avg_precision,
            'recall@k': avg_recall,
            'n_users': len(precisions)
        }
    
    def evaluate_all(self, k=10):
        """
        Đánh giá tất cả models
        
        Args:
            k: Top K cho Precision@K và Recall@K
        
        Returns:
            dict với tất cả kết quả
        """
        print("\n" + "=" * 80)
        print("EVALUATING ALL MODELS")
        print("=" * 80)
        
        results = {}
        
        # 1. Collaborative: RMSE, MAE
        print("\n" + "-" * 80)
        collab_results = self.evaluate_collaborative_rmse_mae()
        results['collaborative_rmse_mae'] = collab_results
        
        # 2. Content-Based: Precision@K, Recall@K
        print("\n" + "-" * 80)
        content_results = self.evaluate_precision_recall_at_k(
            self.content_recommender,
            "Content-Based",
            k=k
        )
        results['content_based'] = content_results
        
        # 3. Collaborative: Precision@K, Recall@K
        print("\n" + "-" * 80)
        # Dùng collaborative recommender từ hybrid (đã load sẵn)
        collab_results_pr = self.evaluate_precision_recall_at_k(
            self.hybrid_recommender.collab_recommender,
            "Collaborative",
            k=k
        )
        results['collaborative_pr'] = collab_results_pr
        
        # 4. Hybrid: Precision@K, Recall@K
        print("\n" + "-" * 80)
        hybrid_results = self.evaluate_precision_recall_at_k(
            self.hybrid_recommender,
            "Hybrid",
            k=k
        )
        results['hybrid'] = hybrid_results
        
        # Tổng kết
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\n📊 Collaborative Filtering (Rating Prediction):")
        print(f"   RMSE: {results['collaborative_rmse_mae']['rmse']:.4f}")
        print(f"   MAE:  {results['collaborative_rmse_mae']['mae']:.4f}")
        
        print(f"\n📊 Content-Based (Top-{k} Recommendations):")
        print(f"   Precision@{k}: {results['content_based']['precision@k']:.4f}")
        print(f"   Recall@{k}:    {results['content_based']['recall@k']:.4f}")
        
        print(f"\n📊 Collaborative (Top-{k} Recommendations):")
        print(f"   Precision@{k}: {results['collaborative_pr']['precision@k']:.4f}")
        print(f"   Recall@{k}:    {results['collaborative_pr']['recall@k']:.4f}")
        
        print(f"\n📊 Hybrid (Top-{k} Recommendations):")
        print(f"   Precision@{k}: {results['hybrid']['precision@k']:.4f}")
        print(f"   Recall@{k}:    {results['hybrid']['recall@k']:.4f}")
        
        return results

def main():
    """Chạy evaluation"""
    print("\n" + "=" * 80)
    print("RECOMMENDATION MODELS EVALUATION")
    print("=" * 80)
    print("\n⚠️  Lưu ý: Quá trình này sẽ mất vài phút để:")
    print("   1. Load models")
    print("   2. Train SVD trên train set")
    print("   3. Evaluate trên test set")
    print("=" * 80)
    
    # Khởi tạo evaluator
    evaluator = RecommendationEvaluator(test_size=0.2, random_state=42)
    
    # Đánh giá tất cả
    results = evaluator.evaluate_all(k=10)
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    results = main()

