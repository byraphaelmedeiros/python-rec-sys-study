import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

if __name__ == '__main__':
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    frame = pd.read_csv('model_resources/ml-100k/u.data', sep='\t', names=columns)
    print(frame.head())

    columns = ['item_id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action',
               'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
               'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    movies = pd.read_csv('model_resources/ml-100k/u.item', sep='|', names=columns, encoding='latin-1')
    print(movies.head())

    combined_movies_data = pd.merge(frame, movies, on='item_id')
    print(combined_movies_data.head())

    print(combined_movies_data.groupby('item_id')['rating'].count().sort_values(ascending=False).head())

    print(movies[movies['item_id'] == 50])

    filter = combined_movies_data['item_id'] == 50
    print(combined_movies_data[filter]['movie title'].unique())

    # utility matrix
    rating_crosstab = combined_movies_data.pivot_table(values='rating', index='user_id', columns='movie title',
                                                       fill_value=0)
    print(rating_crosstab.head())

    # transposing the matrix
    print(rating_crosstab.shape)

    x = rating_crosstab.values.T
    print(x.shape)

    # decomposing the matrix
    SVD = TruncatedSVD(n_components=12, random_state=17)
    resultant_matrix = SVD.fit_transform(x)
    print(resultant_matrix.shape)

    # generating a correlation matrix
    corr_mat = np.corrcoef(resultant_matrix)
    print(corr_mat.shape)

    # isolating star wars from correlation matrix
    movies_names = rating_crosstab.columns
    movies_list = list(movies_names)

    start_wars = movies_list.index('Star Wars (1977)')
    print(start_wars)

    corr_start_wars = corr_mat[start_wars]
    print(corr_start_wars.shape)

    # recommending highly correlated movie
    highly_corr_movies_9 = list(movies_names[(corr_start_wars < 1.0) & (corr_start_wars > 0.9)])
    print(highly_corr_movies_9)

    highly_corr_movies_95 = list(movies_names[(corr_start_wars < 1.0) & (corr_start_wars > 0.95)])
    print(highly_corr_movies_95)
