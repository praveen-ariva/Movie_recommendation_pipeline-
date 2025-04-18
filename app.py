import os
from flask import Flask, render_template, request, redirect, url_for, flash, session # type: ignore
import pandas as pd
from sqlalchemy import create_engine, text
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import numpy as np
from models.recommendation_engine import MovieRecommender

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'movie_recommendation_secret_key'  # For session and flash messages

# Database connection
DB_PATH = os.path.join(os.path.dirname(__file__), 'movie_recommendation.db')
DB_URI = f"sqlite:///{DB_PATH}"
engine = create_engine(DB_URI)

# Initialize recommender
recommender = MovieRecommender(DB_URI)
recommender.build_models()

# Helper function to create plots
def create_plot(plt_obj):
    img = BytesIO()
    plt_obj.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url

@app.route('/')
def index():
    # Check if user is logged in
    user_id = session.get('user_id')
    
    # Get popular movies for non-logged in users
    popular_movies = recommender.get_popular_movies(10)
    
    # Generate rating distribution chart
    plt.figure(figsize=(10, 6))
    with engine.connect() as conn:
        ratings_data = pd.read_sql("SELECT rating FROM ratings", conn)
    
    sns.histplot(ratings_data['rating'], bins=10, kde=True)
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    ratings_chart = create_plot(plt)
    
    # Generate genre distribution chart
    plt.figure(figsize=(12, 8))
    with engine.connect() as conn:
        genres_data = pd.read_sql("SELECT genre, COUNT(*) as count FROM movie_genres GROUP BY genre ORDER BY count DESC LIMIT 10", conn)
    
    sns.barplot(x='count', y='genre', data=genres_data)
    plt.title('Top 10 Movie Genres')
    plt.xlabel('Count')
    plt.ylabel('Genre')
    genres_chart = create_plot(plt)
    
    return render_template('index.html', 
                           user_id=user_id, 
                           popular_movies=popular_movies.to_dict('records'),
                           ratings_chart=ratings_chart,
                           genres_chart=genres_chart)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        
        # Validate user exists in the database
        with engine.connect() as conn:
            user_exists = conn.execute(text("SELECT 1 FROM ratings WHERE userId = :user_id LIMIT 1"), 
                                       {"user_id": user_id}).fetchone()
        
        if user_exists:
            session['user_id'] = int(user_id)
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('User ID not found in the database!', 'danger')
    
    # Get a list of valid user IDs to help testing
    with engine.connect() as conn:
        user_ids = conn.execute(text("SELECT DISTINCT userId FROM ratings ORDER BY userId LIMIT 20")).fetchall()
    
    return render_template('login.html', user_ids=[u[0] for u in user_ids])

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/recommendations')
def recommendations():
    user_id = session.get('user_id')
    
    if not user_id:
        flash('Please login to view personalized recommendations.', 'warning')
        return redirect(url_for('login'))
    
    # Get different types of recommendations
    popular_recs = recommender.get_popular_movies(10)
    collaborative_recs = recommender.get_collaborative_recommendations(user_id, 10)
    hybrid_recs = recommender.get_hybrid_recommendations(user_id, 10)
    
    # Get user's top rated movies
    with engine.connect() as conn:
        top_rated = pd.read_sql(text("""
            SELECT r.movieId, m.title, r.rating, m.genres
            FROM ratings r
            JOIN movies m ON r.movieId = m.movieId
            WHERE r.userId = :user_id
            ORDER BY r.rating DESC
            LIMIT 5
        """), conn, params={"user_id": user_id})
    
    # If we have top rated movies, get content-based recommendations for the first one
    content_recs = pd.DataFrame()
    if not top_rated.empty:
        top_movie_id = top_rated.iloc[0]['movieId']
        content_recs = recommender.get_content_based_recommendations(top_movie_id, 10)
    
    return render_template('recommendations.html', 
                          user_id=user_id,
                          popular_recs=popular_recs.to_dict('records'),
                          collaborative_recs=collaborative_recs.to_dict('records') if not collaborative_recs.empty else [],
                          content_recs=content_recs.to_dict('records') if not content_recs.empty else [],
                          hybrid_recs=hybrid_recs.to_dict('records') if not hybrid_recs.empty else [],
                          top_rated=top_rated.to_dict('records') if not top_rated.empty else [])

@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    # Get movie details
    with engine.connect() as conn:
        movie = pd.read_sql(text("""
            SELECT m.*, 
                   COUNT(r.rating) as num_ratings, 
                   AVG(r.rating) as average_rating
            FROM movies m
            LEFT JOIN ratings r ON m.movieId = r.movieId
            WHERE m.movieId = :movie_id
            GROUP BY m.movieId
        """), conn, params={"movie_id": movie_id}).iloc[0]
        
        # Get genres
        genres = pd.read_sql(text("""
            SELECT genre FROM movie_genres
            WHERE movieId = :movie_id
        """), conn, params={"movie_id": movie_id})
        
        # Get ratings distribution
        ratings_dist = pd.read_sql(text("""
            SELECT rating, COUNT(*) as count
            FROM ratings
            WHERE movieId = :movie_id
            GROUP BY rating
            ORDER BY rating
        """), conn, params={"movie_id": movie_id})
    
    # Create ratings distribution chart
    plt.figure(figsize=(8, 6))
    if not ratings_dist.empty:
        sns.barplot(x='rating', y='count', data=ratings_dist)
        plt.title(f'Rating Distribution for {movie["title"]}')
        plt.xlabel('Rating')
        plt.ylabel('Count')
    else:
        plt.text(0.5, 0.5, "No ratings available", horizontalalignment='center', verticalalignment='center')
    ratings_chart = create_plot(plt)
    
    # Get similar movies (content-based)
    similar_movies = recommender.get_content_based_recommendations(movie_id, 6)
    
    return render_template('movie_details.html',
                          movie=movie.to_dict(),
                          genres=genres['genre'].tolist(),
                          ratings_chart=ratings_chart,
                          similar_movies=similar_movies.to_dict('records') if not similar_movies.empty else [])

@app.route('/search', methods=['GET', 'POST'])
def search():
    search_term = request.args.get('term', '')
    
    if search_term:
        # Search for movies
        with engine.connect() as conn:
            results = pd.read_sql(text("""
                SELECT movieId, title, genres, avg_rating, rating_count
                FROM movies
                WHERE title LIKE :search_term
                ORDER BY rating_count DESC
                LIMIT 50
            """), conn, params={"search_term": f"%{search_term}%"})
    else:
        results = pd.DataFrame()
    
    return render_template('search.html', 
                          search_term=search_term,
                          results=results.to_dict('records') if not results.empty else [])

@app.route('/analytics')
def analytics():
    # Generate analytics for the dashboard
    
    # 1. Top rated movies (min 100 ratings)
    with engine.connect() as conn:
        top_rated = pd.read_sql("""
            SELECT m.movieId, m.title, m.avg_rating, m.rating_count
            FROM movies m
            WHERE m.rating_count >= 100
            ORDER BY m.avg_rating DESC
            LIMIT 10
        """, conn)
    
    # 2. Most popular movies by number of ratings
    with engine.connect() as conn:
        most_popular = pd.read_sql("""
            SELECT m.movieId, m.title, m.avg_rating, m.rating_count
            FROM movies m
            ORDER BY m.rating_count DESC
            LIMIT 10
        """, conn)
    
    # 3. Genre distribution
    with engine.connect() as conn:
        genre_dist = pd.read_sql("""
            SELECT genre, COUNT(*) as count
            FROM movie_genres
            GROUP BY genre
            ORDER BY count DESC
        """, conn)
    
    # Create genre pie chart
    plt.figure(figsize=(10, 10))
    if len(genre_dist) > 10:
        # Combine small genres into "Other"
        other_count = genre_dist['count'][10:].sum()
        top_genres = genre_dist.iloc[:10].copy()
        top_genres = pd.concat([top_genres, pd.DataFrame([{'genre': 'Other', 'count': other_count}])])
        labels = top_genres['genre']
        sizes = top_genres['count']
    else:
        labels = genre_dist['genre']
        sizes = genre_dist['count']
        
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Movie Genre Distribution')
    genres_pie = create_plot(plt)
    
    # 4. Ratings over time
    with engine.connect() as conn:
        # Convert Unix timestamp to year
        ratings_time = pd.read_sql("""
            SELECT strftime('%Y', datetime(timestamp, 'unixepoch')) as year,
                   COUNT(*) as count
            FROM ratings
            GROUP BY year
            ORDER BY year
        """, conn)
    
    # Create ratings over time line chart
    plt.figure(figsize=(12, 6))
    if not ratings_time.empty:
        sns.lineplot(x='year', y='count', data=ratings_time, marker='o')
        plt.title('Number of Ratings by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Ratings')
        plt.xticks(rotation=45)
    ratings_time_chart = create_plot(plt)
    
    return render_template('analytics.html',
                          top_rated=top_rated.to_dict('records'),
                          most_popular=most_popular.to_dict('records'),
                          genres_pie=genres_pie,
                          ratings_time_chart=ratings_time_chart)

if __name__ == '__main__':
    app.run(debug=True)