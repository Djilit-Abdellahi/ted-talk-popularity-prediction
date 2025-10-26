TED Talk Popularity Prediction

This project aims to predict the viewership (popularity) of TED Talks using machine learning regression techniques. It leverages metadata, speaker information, and engagement features to build a model that forecasts view counts and identifies the key factors driving popularity.

Project Goal

Build a robust regression model to predict the total views of a TED Talk and uncover the most influential features contributing to high viewership.

Dataset

The analysis uses a dataset containing information on over 2,500 TED talks, including:

Metadata: Title, Description, Duration, Event Type (TED/TEDx), Filming & Publication Dates, Languages Available.

Speaker Info: Main Speaker Name, Occupation, Number of Speakers.

Engagement: Comments, Ratings, Related Talks.

Target Variable: views.

The dataset used is G-A.csv. (Note: If this dataset is publicly available, add a link here. If it's too large for GitHub, mention that it needs to be obtained separately).

Project Workflow

Data Loading & Initial Processing:

Load the dataset using Pandas.

Convert timestamp columns (film_date, published_date) to datetime objects.

Parse string representations of lists/dictionaries (tags, ratings, related_talks).

Handle missing values (specifically dropping rows with missing speaker_occupation).

Exploratory Data Analysis (EDA):

Analyze distributions of key numerical features (views, comments, duration).

Investigate correlations between numerical features.

Explore trends related to speakers (frequency, average views), occupations, publication year, event type, language availability, and popular tags (using WordCloud).

Feature Engineering:

related_views: Calculated the average views of related talks.

avg_speaker_view: Calculated the average views for each main speaker across all their talks.

weighted_occupation: Assigned a rank/weight to each speaker occupation based on its average view count.

published_year: Extracted from the published_date.

One-Hot Encoding: Applied to the event column (TED vs. TEDx) and the top 10 most frequent tags.

Data Cleaning & Preprocessing for Modeling:

Removed unnecessary or redundant columns (name, title, url, description, ratings, main_speaker, speaker_occupation, original tags, date columns).

Treated outliers in numerical features by capping them at 1.5 * IQR.

Feature Selection:

Evaluated feature importance using Forward Selection, Lasso Regression (L1 Regularization), and Random Forest Feature Importance.

Selected the most relevant features based on these methods (primarily avg_speaker_view, comments, languages, weighted_occupation, related_views, duration, published_year).

Removed low-importance tag features.

Model Building & Evaluation:

Split data into training (80%) and testing (20%) sets.

Trained and evaluated several regression models:

Lasso Regression

Ridge Regression

Random Forest Regressor

Gradient Boosting Regressor

Support Vector Regressor (SVR)

Bagging Regressor (with Decision Tree base)

Used R² Score, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) as evaluation metrics.

Hyperparameter Tuning:

Applied GridSearchCV to optimize hyperparameters for Lasso, Ridge, Random Forest, Gradient Boosting, and SVR models to improve performance.

Model Comparison & Conclusion:

Compared the performance of tuned models on the test set.

Identified tree-based models (Random Forest, Gradient Boosting) as the best performers.

Concluded that engineered features like avg_speaker_view and weighted_occupation were highly influential.

Key Findings & Results

Most Influential Features: avg_speaker_view (average views of the speaker), comments, languages, and weighted_occupation showed the strongest correlation with view counts.

Best Performing Models: Tree-based ensemble methods like Gradient Boosting and Random Forest achieved the highest R² scores (around 0.86 - 0.88 on the test set after tuning), significantly outperforming linear models and SVR.

Impact of Feature Engineering: Creating new features based on speaker history and occupation drastically improved model performance compared to using only the raw features.

Outlier Treatment: Capping outliers was necessary to stabilize model training.

Technologies Used

Python: Core programming language.

Pandas: Data manipulation and analysis.

NumPy: Numerical computations.

Scikit-learn:

Model Building (Lasso, Ridge, RandomForestRegressor, GradientBoostingRegressor, SVR, BaggingRegressor, DecisionTreeRegressor)

Preprocessing (StandardScaler, train_test_split)

Feature Selection (LassoCV)

Model Evaluation (mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error)

Hyperparameter Tuning (GridSearchCV)

Matplotlib & Seaborn: Data visualization.

WordCloud & Squarify: Visualizing tag and speaker frequency.

Jupyter Notebook: Development environment.

Ensure Dataset: Make sure the G-A.csv dataset file is in the repository directory (or update the path in the notebook).

Run the Jupyter Notebook:

jupyter notebook Projet_TedTalk_FINAL.ipynb


Execute the cells sequentially to reproduce the analysis and model training.

Future Work

Incorporate NLP analysis on talk transcripts or descriptions.

Explore deep learning models (Neural Networks).

Add visual features if available (e.g., thumbnail analysis).

Deploy the best model as an API or web application.
