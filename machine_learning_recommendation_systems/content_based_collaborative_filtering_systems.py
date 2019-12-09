import pandas as pd

from sklearn.neighbors import NearestNeighbors

if __name__ == '__main__':
    cars = pd.read_csv('content_resources/mtcars.csv')
    cars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
    print(cars.head())

    t = [15, 300, 160, 3.2]

    x = cars.ix[:, (1, 3, 4, 6)].values
    print(x[0:5])

    nbrs = NearestNeighbors(n_neighbors=1).fit(x)
    print(nbrs.kneighbors([t]))

    print(cars)