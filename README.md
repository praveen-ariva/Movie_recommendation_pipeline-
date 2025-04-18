# Movie Recommendation Pipeline

A complete data engineering project that demonstrates ETL processes, database design, recommendation algorithms, and data visualization through a web dashboard.

## Project Overview

This project builds a complete movie recommendation system with the following components:

1. **ETL Pipeline**: Extract, transform, and load movie data into a database
2. **Database**: SQLite database for storing movie, user, and rating data
3. **Recommendation Engine**: Multiple recommendation algorithms (content-based, collaborative filtering, hybrid)
4. **Web Dashboard**: Flask-based web application to display recommendations and analytics

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/movie-recommendation-pipeline.git
   cd movie-recommendation-pipeline
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the MovieLens dataset:
   - Visit [MovieLens](https://grouplens.org/datasets/movielens/)
   - Download the "MovieLens Latest Small" dataset (~1MB) for testing
   - Extract the contents to the `data/ml-latest-small` directory

### Running the ETL Pipeline

1. Run the ETL script to process the MovieLens data and load it into the database:
   ```
   python etl/etl_pipeline.py
   ```

2. This will:
   - Extract data from the CSV files
   - Transform the data (extract year from title, calculate average ratings, etc.)
   - Load the transformed data into the SQLite database

### Running the Recommendation Engine

1. The recommendation engine is built into the web application, but you can test it separately:
   ```
   python models/recommendation_engine.py
   ```

2. This will:
   - Load data from the database
   - Build content-based and collaborative filtering models
   - Generate sample recommendations

### Running the Web Dashboard

1. Start the Flask web server:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. You can now:
   - Browse popular movies
   - Login with a user ID from the MovieLens dataset
   - View personalized recommendations
   - Search for movies
   - Explore analytics

## Project Structure

```
movie_recommendation_pipeline/
│
├── data/                          # Data directory
│   └── ml-latest-small/           # MovieLens dataset
│
├── etl/                           # ETL pipeline code
│   └── etl_pipeline.py            # ETL script
│
├── models/                        # Recommendation models
│   └── recommendation_engine.py   # Recommendation engine
│
├── static/                        # Static files for Flask
│   ├── css/
│   ├── js/
│   └── img/
│
├── templates/                     # HTML templates
│   ├── base.html                  # Base template
│   ├── index.html                 # Home page
│   └── ...                        # Other page templates
│
├── app.py                         # Flask application
├── schema.sql                     # Database schema
└── requirements.txt               # Dependencies
```

## Key Technical Components

### ETL Pipeline Features
- Data extraction from CSV files
- Data cleaning and transformation
- Calculation of derived metrics
- Database loading with relationship preservation
- Logging and error handling

### Recommendation Engine Features
- Content-based filtering using movie genres
- Collaborative filtering using user-item matrix
- Hybrid recommendations
- Popular movie recommendations
- Nearest neighbors algorithm for finding similar users/items

### Dashboard Features
- User authentication
- Personalized movie recommendations
- Movie search functionality
- Detailed movie information
- Data visualizations and analytics

## Data Engineering Concepts Applied

- **Data Extraction**: Reading from structured CSV files
- **Data Transformation**: Cleaning, enrichment, normalization
- **Data Loading**: Efficient database loading with proper indexing
- **Data Modeling**: Relational database design with appropriate relationships
- **Machine Learning**: Recommendation algorithms implementation
- **Data Visualization**: Charts and graphs for analytics
- **Web Development**: Front-end and back-end integration

## Future Enhancements

- Use a production-grade database (PostgreSQL, MySQL)
- Implement a scheduled ETL process using Airflow
- Add real-time recommendations using a message queue
- Deploy using Docker containers
- Add user registration and personalized watchlists
- Implement A/B testing for recommendation algorithms

## License

This project uses the MovieLens dataset which is distributed for non-commercial use.