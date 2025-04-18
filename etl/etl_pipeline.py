import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("etl_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MovieETL:
    def __init__(self, data_path, db_connection_string):
        """
        Initialize the ETL pipeline
        
        Args:
            data_path (str): Path to the directory containing the MovieLens data files
            db_connection_string (str): SQLAlchemy database connection string
        """
        self.data_path = data_path
        self.db_connection_string = db_connection_string
        self.engine = create_engine(db_connection_string)
        
    def extract(self):
        """Extract data from CSV files"""
        logger.info("Starting data extraction")
        
        try:
            # Load movies data
            movies_file = os.path.join(self.data_path, 'movies.csv')
            self.movies_df = pd.read_csv(movies_file)
            logger.info(f"Loaded {len(self.movies_df)} movies")
            
            # Load ratings data
            ratings_file = os.path.join(self.data_path, 'ratings.csv')
            self.ratings_df = pd.read_csv(ratings_file)
            logger.info(f"Loaded {len(self.ratings_df)} ratings")
            
            # Load tags data if it exists
            tags_file = os.path.join(self.data_path, 'tags.csv')
            if os.path.exists(tags_file):
                self.tags_df = pd.read_csv(tags_file)
                logger.info(f"Loaded {len(self.tags_df)} tags")
            else:
                self.tags_df = None
                logger.info("No tags file found")
                
            return True
            
        except Exception as e:
            logger.error(f"Error during extraction: {str(e)}")
            return False
    
    def transform(self):
        """Transform the extracted data"""
        logger.info("Starting data transformation")
        
        try:
            # 1. Extract year from title and create a separate column
            logger.info("Extracting year from movie titles")
            self.movies_df['year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)$')
            self.movies_df['title'] = self.movies_df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
            
            # 2. Split genres into a list
            logger.info("Processing movie genres")
            self.movies_df['genres_list'] = self.movies_df['genres'].str.split('|')
            
            # 3. Calculate average rating for each movie
            logger.info("Calculating average ratings")
            avg_ratings = self.ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
            avg_ratings.columns = ['movieId', 'avg_rating', 'rating_count']
            
            # 4. Merge average ratings with movies
            self.movies_enriched = pd.merge(self.movies_df, avg_ratings, on='movieId', how='left')
            self.movies_enriched['avg_rating'].fillna(0, inplace=True)
            self.movies_enriched['rating_count'].fillna(0, inplace=True)
            
            # 5. Create a timestamp column for when this data was processed
            process_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.movies_enriched['processed_at'] = process_date
            self.ratings_df['processed_at'] = process_date
            
            logger.info("Data transformation completed")
            return True
            
        except Exception as e:
            logger.error(f"Error during transformation: {str(e)}")
            return False
    
    def load(self):
        """Load the transformed data into the database"""
        logger.info("Starting data loading")
        
        try:
            # Create a connection to the database
            logger.info(f"Connecting to database: {self.db_connection_string}")
            
            # Load movies table
            logger.info("Loading movies data")
            self.movies_enriched.drop(columns=['genres_list'], errors='ignore').to_sql(
                'movies', 
                self.engine, 
                if_exists='replace', 
                index=False
            )
            
            # Load ratings table
            logger.info("Loading ratings data")
            self.ratings_df.to_sql(
                'ratings', 
                self.engine, 
                if_exists='replace', 
                index=False,
                chunksize=10000  # Load in chunks to handle large datasets
            )
            
            # Load tags table if it exists
            if self.tags_df is not None:
                logger.info("Loading tags data")
                self.tags_df['processed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.tags_df.to_sql(
                    'tags', 
                    self.engine, 
                    if_exists='replace', 
                    index=False
                )
            
            # Create movie_genres table (for normalized data model)
            logger.info("Creating normalized genre tables")
            genres_data = []
            
            for _, row in self.movies_df.iterrows():
                movie_id = row['movieId']
                genres = row['genres'].split('|')
                for genre in genres:
                    if genre != '(no genres listed)':
                        genres_data.append({'movieId': movie_id, 'genre': genre})
            
            genres_df = pd.DataFrame(genres_data)
            genres_df.to_sql('movie_genres', self.engine, if_exists='replace', index=False)
            
            logger.info("Data loading completed")
            return True
            
        except Exception as e:
            logger.error(f"Error during loading: {str(e)}")
            return False
    
    def run_pipeline(self):
        """Run the complete ETL pipeline"""
        logger.info("Starting ETL pipeline")
        
        extract_success = self.extract()
        if not extract_success:
            logger.error("Extraction failed, stopping pipeline")
            return False
        
        transform_success = self.transform()
        if not transform_success:
            logger.error("Transformation failed, stopping pipeline")
            return False
        
        load_success = self.load()
        if not load_success:
            logger.error("Loading failed, stopping pipeline")
            return False
        
        logger.info("ETL pipeline completed successfully")
        return True


if __name__ == "__main__":
    # Example usage
    data_path = "./ml-latest-small"
    
    # Use SQLite for simplicity
    db_connection_string = "sqlite:///movie_recommendation.db"
    
    # Create and run the ETL pipeline
    etl = MovieETL(data_path, db_connection_string)
    success = etl.run_pipeline()
    
    if success:
        print("ETL process completed successfully!")
    else:
        print("ETL process failed. Check logs for details.")