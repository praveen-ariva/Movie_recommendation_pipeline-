-- Database Schema for Movie Recommendation System

-- Movies table
CREATE TABLE IF NOT EXISTS movies (
    movieId INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    genres TEXT,
    year INTEGER,
    avg_rating REAL DEFAULT 0,
    rating_count INTEGER DEFAULT 0,
    processed_at TIMESTAMP
);

-- Create index on title for faster searches
CREATE INDEX IF NOT EXISTS idx_movies_title ON movies(title);

-- Ratings table
CREATE TABLE IF NOT EXISTS ratings (
    ratingId INTEGER PRIMARY KEY AUTOINCREMENT,
    userId INTEGER NOT NULL,
    movieId INTEGER NOT NULL,
    rating REAL NOT NULL,
    timestamp INTEGER,
    processed_at TIMESTAMP,
    FOREIGN KEY (movieId) REFERENCES movies(movieId)
);

-- Create indexes for faster joins and queries
CREATE INDEX IF NOT EXISTS idx_ratings_userId ON ratings(userId);
CREATE INDEX IF NOT EXISTS idx_ratings_movieId ON ratings(movieId);

-- Tags table
CREATE TABLE IF NOT EXISTS tags (
    tagId INTEGER PRIMARY KEY AUTOINCREMENT,
    userId INTEGER NOT NULL,
    movieId INTEGER NOT NULL,
    tag TEXT NOT NULL,
    timestamp INTEGER,
    processed_at TIMESTAMP,
    FOREIGN KEY (movieId) REFERENCES movies(movieId)
);

-- Create indexes for faster joins and queries
CREATE INDEX IF NOT EXISTS idx_tags_movieId ON tags(movieId);
CREATE INDEX IF NOT EXISTS idx_tags_userId ON tags(userId);

-- Movie genres normalized table (many-to-many)
CREATE TABLE IF NOT EXISTS movie_genres (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    movieId INTEGER NOT NULL,
    genre TEXT NOT NULL,
    FOREIGN KEY (movieId) REFERENCES movies(movieId)
);

-- Create indexes for movie genres
CREATE INDEX IF NOT EXISTS idx_movie_genres_movieId ON movie_genres(movieId);
CREATE INDEX IF NOT EXISTS idx_movie_genres_genre ON movie_genres(genre);

-- User recommendations table
CREATE TABLE IF NOT EXISTS user_recommendations (
    recommendationId INTEGER PRIMARY KEY AUTOINCREMENT,
    userId INTEGER NOT NULL,
    movieId INTEGER NOT NULL,
    score REAL NOT NULL,
    recommendation_type TEXT NOT NULL, -- 'content_based', 'collaborative', 'popular'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (movieId) REFERENCES movies(movieId)
);

-- Create indexes for recommendations
CREATE INDEX IF NOT EXISTS idx_recommendations_userId ON user_recommendations(userId);