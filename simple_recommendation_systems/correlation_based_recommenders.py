import pandas as pd
import numpy as np

if __name__ == '__main__':
    frame = pd.read_csv('correlation_resources/rating_final.csv')
    cuisine = pd.read_csv('correlation_resources/chefmozcuisine.csv')
    geo_data = pd.read_csv('correlation_resources/geoplaces2.csv')

    print(frame.head())
    print(cuisine.head())
    # print(geodata.head())

    places = geo_data[['placeID', 'name']]
    print(places.head())

    rating = pd.DataFrame(frame.groupby('placeID')['rating'].mean())
    print(rating.head())

    rating['rating_count'] = pd.DataFrame(frame.groupby('placeID')['rating'].count())
    print(rating.head())
    print(rating.describe())

    print(rating.sort_values('rating_count', ascending=False).head())

    print(places[places['placeID']==135085])
    print(cuisine[cuisine['placeID'] == 135085])

    # preparing data for analysis
    places_crosstab = pd.pivot_table(data=frame, values='rating', index='userID', columns='placeID')
    print(places_crosstab.head())

    tortas_ratings = places_crosstab[135085]
    print(tortas_ratings[tortas_ratings >= 0])

    # evaluating similarity based on correlation
    similar_to_tortas = places_crosstab.corrwith(tortas_ratings)

    corr_tortas = pd.DataFrame(similar_to_tortas, columns=['PearsonR'])
    corr_tortas.dropna(inplace=True)
    print(corr_tortas.head())

    tortas_corr_summary = corr_tortas.join(rating['rating_count'])
    print(tortas_corr_summary[tortas_corr_summary['rating_count'] >= 10].sort_values('PearsonR', ascending=False)
          .head(10))

    places_corr_tortas = pd.DataFrame([135085, 132754, 135045, 135062, 135028, 135042, 135046], index=np.arange(7),
                                      columns=['placeID'])
    summary = pd.merge(places_corr_tortas, cuisine, on='placeID')
    print(summary)

    print(places[places['placeID']==135046])

    print(cuisine['Rcuisine'].describe())