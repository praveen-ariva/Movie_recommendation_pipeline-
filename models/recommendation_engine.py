import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommendation_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MovieRecommender:
    """Movie recommendation engine that implements multiple recommendation strategies"""
    
    def __init__(self, db_connection_string):
        """
        Initialize the recommendation engine
        
        Args:
            db_connection_string (str): SQLAlchemy database connection string
        """
        self.db_connection_string = db_connection_string
        self.engine = create_engine(db_connection_string)
        self.load_data()
        
    def load_data(self):
        """Load data from the database"""
        logger.info("Loading data from database")
        
        try:
            # Load movies
            self.movies_df = pd.read_sql("SELECT * FROM movies", self.engine)
            logger.info(f"Loaded {len(self.movies_df)} movies")
            
            # Load ratings
            self.ratings_df = pd.read_sql("SELECT * FROM ratings", self.engine)
            logger.info(f"Loaded {len(self.ratings_df)} ratings")
            
            # Load movie genres
            self.genres_df = pd.read_sql("SELECT * FROM movie_genres", self.engine)
            logger.info(f"Loaded {len(self.genres_df)} genre entries")
            
            # Create a list of all users
            self.users = self.ratings_df['userId'].unique()
            logger.info(f"Found {len(self.users)} unique users")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def build_content_based_model(self):
        """
        Build a content-based recommendation model using movie genres
        """
        logger.info("Building content-based recommendation model")
        
        try:
            # Create a string of genres for each movie
            genre_data = self.genres_df.groupby('movieId')['genre'].apply(lambda x: ' '.join(x)).reset_index()
            
            # Merge with movies data
            content_df = pd.merge(self.movies_df[['movieId', 'title']], genre_data, on='movieId', how='left')
            content_df['genre'] = content_df['genre'].fillna('')
            
            # Combine title and genre for better content-based recommendations
            content_df['content'] = content_df['title'] + ' ' + content_df['genre']
            
            # Create TF-IDF vectors for movie content
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(content_df['content'])
            
            # Compute similarity matrix
            self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            # Create a mapping from movie ID to index
            self.indices = pd.Series(content_df.index, index=content_df['movieId']).drop_duplicates()
            
            logger.info("Content-based model built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error building content-based model: {str(e)}")
            return False
    
    def build_collaborative_filtering_model(self):
        """
        Build a collaborative filtering model using user-item ratings matrix
        """
        logger.info("Building collaborative filtering model")
        
        try:
            # Create a user-item matrix
            user_item_matrix = self.ratings_df.pivot(
                index='userId', 
                columns='movieId', 
                values='rating'
            ).fillna(0)
            
            # Convert to sparse matrix for efficiency
            self.user_item_sparse = csr_matrix(user_item_matrix.values)
            
            # Fit KNN model
            self.model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
            self.model_knn.fit(self.user_item_sparse)
            
            # Save mapping from matrix index to userID
            self.user_mapper = dict(zip(range(len(user_item_matrix.index)), user_item_matrix.index))
            self.movie_mapper = dict(zip(range(len(user_item_matrix.columns)), user_item_matrix.columns))
            self.user_inv_mapper = {v: k for k, v in self.user_mapper.items()}
            
            logger.info("Collaborative filtering model built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error building collaborative filtering model: {str(e)}")
            return False
    
    def get_popular_movies(self, n=10):
        """
        Get the most popular movies based on number of ratings and average score
        
        Args:
            n (int): Number of recommendations to return
            
        Returns:
            DataFrame: Dataframe with movie recommendations
        """
        logger.info(f"Getting {n} popular movies")
        
        try:
            # Define popularity as a combination of rating count and average rating
            # Only consider movies with at least 100 ratings
            popular_movies = self.movies_df[self.movies_df['rating_count'] >= 100].copy()
            
            # Calculate popularity score
            popular_movies['popularity_score'] = (
                popular_movies['rating_count'] * 0.7 + 
                popular_movies['avg_rating'] * 30
            )
            
            # Sort by popularity score and return top n
            recommendations = popular_movies.sort_values(
                'popularity_score', 
                ascending=False
            ).head(n)
            
            return recommendations[['movieId', 'title', 'avg_rating', 'rating_count']]
            
        except Exception as e:
            logger.error(f"Error getting popular movies: {str(e)}")
            return pd.DataFrame()
    
    def get_content_based_recommendations(self, movie_id, n=10):
        """
        Get content-based recommendations for a movie
        
        Args:
            movie_id (int): Movie ID to get recommendations for
            n (int): Number of recommendations to return
            
        Returns:
            DataFrame: Dataframe with movie recommendations
        """
        logger.info(f"Getting {n} content-based recommendations for movie {movie_id}")
        
        try:
            # Check if movie_id exists in our dataset
            if movie_id not in self.indices:
                logger.warning(f"Movie ID {movie_id} not found in the dataset")
                return pd.DataFrame()
                
            # Get the index of the movie
            idx = self.indices[movie_id]
            
            # Get the similarity scores for all movies with this one
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            
            # Sort based on similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get the top n most similar movies (excluding the movie itself)
            sim_scores = sim_scores[1:n+1]
            
            # Get the movie indices
            movie_indices = [i[0] for i in sim_scores]
            
            # Get the similarity scores
            similarities = [i[1] for i in sim_scores]
            
            # Get the movie IDs
            recommended_ids = self.movies_df.iloc[movie_indices]['movieId'].tolist()
            
            # Create a result dataframe
            recommendations = self.movies_df[self.movies_df['movieId'].isin(recommended_ids)]
            recommendations['similarity_score'] = similarities
            
            return recommendations[['movieId', 'title', 'avg_rating', 'similarity_score']]
            
        except Exception as e:
            logger.error(f"Error getting content-based recommendations: {str(e)}")
            return pd.DataFrame()
    
    def get_collaborative_recommendations(self, user_id, n=10):
        """
        Get collaborative filtering recommendations for a user
        
        Args:
            user_id (int): User ID to get recommendations for
            n (int): Number of recommendations to return
            
        Returns:
            DataFrame: Dataframe with movie recommendations
        """
        logger.info(f"Getting {n} collaborative recommendations for user {user_id}")
        
        try:
            # Check if user_id exists in our dataset
            if user_id not in self.user_inv_mapper:
                logger.warning(f"User ID {user_id} not found in the dataset")
                return pd.DataFrame()
                
            # Get the index of the user
            user_idx = self.user_inv_mapper[user_id]
            
            # Find similar users
            distances, indices = self.model_knn.kneighbors(
                self.user_item_sparse[user_idx].reshape(1, -1), 
                n_neighbors=11
            )
            
            # First element is the user itself, so skip it
            similar_users = [self.user_mapper[idx] for idx in indices.flatten()[1:]]
            distances = distances.flatten()[1:]
            
            # Get movies that similar users liked but the target user hasn't rated
            target_user_movies = set(self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'])
            
            # Get movies rated highly by similar users
            similar_user_ratings = self.ratings_df[
                (self.ratings_df['userId'].isin(similar_users)) & 
                (self.ratings_df['rating'] >= 4.0)
            ]
            
            # Remove movies already rated by the target user
            new_recommendations = similar_user_ratings[
                ~similar_user_ratings['movieId'].isin(target_user_movies)
            ]
            
            # Group by movie and calculate recommendation score
            # Weight by similarity to user (closer users have smaller distance scores)
            movie_scores = new_recommendations.groupby('movieId').agg({
                'rating': 'mean',
                'userId': 'count'
            }).reset_index()
            
            movie_scores['rec_score'] = movie_scores['rating'] * movie_scores['userId']
            movie_scores = movie_scores.sort_values('rec_score', ascending=False).head(n)
            
            # Merge with movie information
            recommendations = pd.merge(
                movie_scores, 
                self.movies_df[['movieId', 'title', 'avg_rating']], 
                on='movieId'
            )
            
            return recommendations[['movieId', 'title', 'avg_rating', 'rec_score']]
            
        except Exception as e:
            logger.error(f"Error getting collaborative recommendations: {str(e)}")
            return pd.DataFrame()
    
    def get_hybrid_recommendations(self, user_id, n=10):
        """
        Get hybrid recommendations (mix of content-based and collaborative)
        
        Args:
            user_id (int): User ID to get recommendations for
            n (int): Number of recommendations to return
            
        Returns:
            DataFrame: Dataframe with movie recommendations
        """
        logger.info(f"Getting {n} hybrid recommendations for user {user_id}")
        
        try:
            # Get collaborative recommendations
            collab_recs = self.get_collaborative_recommendations(user_id, n=n)
            
            # If we couldn't get collaborative recommendations, return popular movies
            if collab_recs.empty:
                logger.info(f"No collaborative recommendations for user {user_id}, returning popular movies")
                return self.get_popular_movies(n=n)
            
            # Get a random movie the user has rated highly
            user_high_ratings = self.ratings_df[
                (self.ratings_df['userId'] == user_id) & 
                (self.ratings_df['rating'] >= 4.0)
            ]
            
            # If the user hasn't rated any movies highly, just return collaborative recommendations
            if user_high_ratings.empty:
                return collab_recs
            
            # Get a random highly-rated movie
            sample_movie_id = user_high_ratings.sample(1)['movieId'].iloc[0]
            
            # Get content-based recommendations for this movie
            content_recs = self.get_content_based_recommendations(sample_movie_id, n=n)
            
            # Combine the two recommendation sets
            hybrid_recs = pd.concat([collab_recs.head(n//2), content_recs.head(n//2)])
            
            # Remove duplicates
            hybrid_recs = hybrid_recs.drop_duplicates(subset=['movieId'])
            
            # If we need more recommendations, get popular movies
            if len(hybrid_recs) < n:
                popular_recs = self.get_popular_movies(n=(n - len(hybrid_recs)))
                # Make sure we don't include movies already in the recommendations
                popular_recs = popular_recs[~popular_recs['movieId'].isin(hybrid_recs['movieId'])]
                hybrid_recs = pd.concat([hybrid_recs, popular_recs]).head(n)
            
            return hybrid_recs
            
        except Exception as e:
            logger.error(f"Error getting hybrid recommendations: {str(e)}")
            return pd.DataFrame()
    
    def save_recommendations_to_db(self, user_id, recommendations, rec_type):
        """
        Save recommendations to the database
        
        Args:
            user_id (int): User ID
            recommendations (DataFrame): Recommendation dataframe
            rec_type (str): Type of recommendation ('content_based', 'collaborative', 'popular', 'hybrid')
            
        Returns:
            bool: Success status
        """
        logger.info(f"Saving {len(recommendations)} {rec_type} recommendations for user {user_id}")
        
        try:
            # Prepare data for insertion
            now = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Get score column based on recommendation type
            if rec_type == 'content_based':
                score_col = 'similarity_score'
            elif rec_type == 'collaborative' or rec_type == 'hybrid':
                score_col = 'rec_score'
            else:  # popular
                score_col = 'popularity_score' if 'popularity_score' in recommendations.columns else 'avg_rating'
            
            # Create recommendations dataframe
            recs_to_save = pd.DataFrame({
                'userId': [user_id] * len(recommendations),
                'movieId': recommendations['movieId'],
                'score': recommendations[score_col] if score_col in recommendations.columns else 0,
                'recommendation_type': rec_type,
                'created_at': now
            })
            
            # Save to database
            recs_to_save.to_sql('user_recommendations', self.engine, if_exists='append', index=False)
            
            logger.info(f"Successfully saved recommendations for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving recommendations: {str(e)}")
            return False
    
    def build_models(self):
        """Build all recommendation models"""
        content_success = self.build_content_based_model()
        collab_success = self.build_collaborative_filtering_model()
        
        return content_success and collab_success
    
    def generate_recommendations_for_all_users(self, n=10):
        """
        Generate and save recommendations for all users
        
        Args:
            n (int): Number of recommendations per user
            
        Returns:
            bool: Success status
        """
        logger.info(f"Generating recommendations for all {len(self.users)} users")
        
        try:
            for user_id in self.users:
                # Generate different types of recommendations
                popular_recs = self.get_popular_movies(n=n)
                self.save_recommendations_to_db(user_id, popular_recs, 'popular')
                
                collab_recs = self.get_collaborative_recommendations(user_id, n=n)
                if not collab_recs.empty:
                    self.save_recommendations_to_db(user_id, collab_recs, 'collaborative')
                
                hybrid_recs = self.get_hybrid_recommendations(user_id, n=n)
                if not hybrid_recs.empty:
                    self.save_recommendations_to_db(user_id, hybrid_recs, 'hybrid')
            
            logger.info("Successfully generated recommendations for all users")
            return True
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return False


if __name__ == "__main__":
    # Example usage
    db_connection_string = "sqlite:///movie_recommendation.db"
    
    # Create and initialize the recommender
    recommender = MovieRecommender(db_connection_string)
    
    # Build all models
    recommender.build_models()
    
    # Generate some example recommendations
    print("\nPopular movies:")
    popular = recommender.get_popular_movies(10)
    print(popular)
    
    print("\nContent-based recommendations for movie ID 1 (Toy Story):")
    content_recs = recommender.get_content_based_recommendations(1, 5)
    print(content_recs)
    
    print("\nCollaborative recommendations for user ID it:")
    collab_recs = recommender.get_collaborative_recommendations(1, 5)
    print(collab_recs)
    
    print("\nHybrid recommendations for user ID 1:")
    hybrid_recs = recommender.get_hybrid_recommendations(1, 5)
    print(hybrid_recs)
    
    # Generate recommendations for all users
    # recommender.generate_recommendations_for_all_users()