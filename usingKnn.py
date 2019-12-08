import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class recommendationClassKnn:

    def __init__(self):
        # dataset books,student,rating
        self.books = pd.read_csv('booksForExcel.csv', error_bad_lines=False, encoding="latin-1")
        # columns of books dataset
        self.books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher']
        self.users = pd.read_csv('student.csv', error_bad_lines=False, encoding="latin-1")
        # columns of student dataset
        self.users.columns = ['userID', 'Name', 'Age', 'Interest']
        self.ratings = pd.read_csv('ratings.csv', error_bad_lines=False, encoding="latin-1")
        # columns of rating dataset
        self.ratings.columns = ['userID', 'ISBN', 'bookRating']

    def show(self):
        print(self.books)
        print(self.users)
        print(self.ratings)

    def shapeOfData(self):
        # shape of rating dataset,gives the dimension of dataset i.e the number of rows and columns
        print(self.ratings.shape)
        # list of rating columns
        print(list(self.ratings.columns))
        print(self.books.shape)
        # list of books columns
        print(list(self.books.columns))
        print(self.users.shape)
        # list of student columns￼
        print(list(self.users.columns))

    def diagram(self):
        # # rating distribution using histogram
        plt.rc("font", size=15)
        self.ratings.bookRating.value_counts(sort=False).plot(kind='bar')
        plt.title('Rating Distribution\n')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.savefig('system1.png', bbox_inches='tight')
        plt.show()

        # student age distribution using histogram
        self.users.Age.hist(bins=[18, 20, 22, 24, 26, 28, 30, 32, 40])
        plt.title('Age Distribution\n')
        plt.xlabel('Age')
        plt.ylabel('Count')
        plt.savefig('system2.png', bbox_inches='tight')
        plt.show()

        #     -------------------------------------------------------------------------

        # boxplot
        self.ratings.boxplot(column=['bookRating'], grid=False)
        self.users.boxplot(column=['Age'])

        # Draw a vertical boxplot grouped
        # by a categorical variable:
        sns.set_style("whitegrid")
        sns.boxplot(y='bookRating', data=self.ratings)
        sns.boxplot(y='Age', data=self.users)

    def recommendation(self):
        # recommendation based on rating count
        rating_count = pd.DataFrame(self.ratings.groupby('ISBN')['bookRating'].count())
        print(rating_count)
        # sorting of the counts of rating to get the highest rated books
        rating_count.sort_values('bookRating', ascending=False).head()
        xy = rating_count.sort_values('bookRating', ascending=False).head(5)
        print(xy)

        # books details of first 5 book which received highest rating by students
        most_rated_books = pd.DataFrame(['978-8120349391', '978-0198070887', '978-9351341741',
                                         '978-0198083542', '978-9351343257'], index=np.arange(5), columns=['ISBN'])
        most_rated_books_summary = pd.merge(most_rated_books, self.books, on='ISBN')
        print(most_rated_books_summary)

        # recommendations based on correlations
        # here Pearson correlation coefficient used to measure the linear correlation between
        # two variable --- the ratings for two books
        # fetch the average rating and the count of rating each book received
        average_rating = pd.DataFrame(self.ratings.groupby('ISBN')['bookRating'].mean())
        print(average_rating)
        average_rating['ratingCount'] = pd.DataFrame(
            self.ratings.groupby('ISBN')['bookRating'].count())
        average_rating.sort_values('ratingCount', ascending=False).head(10)
        # here the main disadvantage is that, the book which got highest number of rating
        # has the rating average is low
        # observation-- in this dataset the book that received the most rating counts was not highly
        # rated at all. So if we are going to use recommendation based on
        # rating counts,we would definitely make mistake or wrong recommendation.

        # to ensure statistical significance,

        # student who rate books and their count >=3
        counts1 = self.ratings['userID'].value_counts()
        print(counts1)
        ratings = self.ratings[self.ratings['userID'].isin(counts1[counts1 >= 1].index)]
        print(ratings)
        counts = ratings['bookRating'].value_counts()
        print(counts)
        # rating of book > 2
        ratings = ratings[ratings['bookRating'].isin(counts[counts >= 1].index)]
        print(ratings)

        # ----------------------------------------------------------------------------------
        # now recommend book using KNN algorithm
        # Collaborative Filtering Using k-Nearest Neighbors (kNN)
        # kNN is a machine learning algorithm to find clusters of similar users based on common book ratings,
        # and make predictions using the average rating of top-k nearest neighbors.
        #  For example, we first present ratings in a matrix with the matrix having one row for each item (book) and one column for each user,

        # merging of rating abd books dataset based on ISBN
        combine_book_rating = pd.merge(self.ratings, self.books, on='ISBN')
        columns = ['yearOfPublication', 'publisher']
        # dropping these columns because these are not required
        print(columns)
        combine_book_rating = combine_book_rating.drop(columns, axis=1)
        combine_book_rating.head(20)

        # We then group by book titles and create a new column for total rating count.
        combine_book_rating = combine_book_rating.dropna(axis=0, subset=['bookTitle'])
        book_ratingCount = (combine_book_rating.groupby(by=['bookTitle'])['bookRating'].count().reset_index().rename(
            columns={'bookRating': 'totalRatingCount'})[['bookTitle', 'totalRatingCount']])
        book_ratingCount.head(10)

        # We combine the rating data with the total rating count data,
        #  this gives us exactly what we need to find out which books are popular and filter out lesser-known books.

        rating_with_totalRatingCount = combine_book_rating.merge(
            book_ratingCount, left_on='bookTitle', right_on='bookTitle', how='left')
        rating_with_totalRatingCount.head(10)

        pd.set_option('display.float_format', lambda x: '%.3f' % x)
        print(book_ratingCount['totalRatingCount'].describe())

        print(book_ratingCount['totalRatingCount'].quantile(np.arange(.9, 1, .01)))

        # threshold value = 5 that is
        # if this is greater than totalRating then suggest
        popularity_threshold = 5
        rating_popular_book = rating_with_totalRatingCount.query(
            'totalRatingCount >= @popularity_threshold')
        rating_popular_book.head(20)

        rating_popular_book.shape
        # popular book with highest rating count
        print(rating_popular_book)

        combined = rating_popular_book.merge(
            self.users, left_on='userID', right_on='userID', how='left')
        print(combined)
        # recommend based on user interest lets take interest
        takeInterest = input("Enter user Interest : ")
        interest_user_rating = combined[combined['Interest'].str.contains(takeInterest)]
        interest_user_rating = interest_user_rating.drop('Age', axis=1)
        interest_user_rating.head(50)

        # now time to apply cosine similarity
        # in this each book is represented in vector
        '''
        Implementing kNN
        We convert our table to a 2D matrix, and fill the missing values with zeros 
        (since we will calculate distances between rating vectors). We then transform the values(ratings)
        of the matrix dataframe into a scipy sparse matrix for more efficient calculations.
        Finding the Nearest Neighbors We use unsupervised algorithms with sklearn.neighbors.
        The algorithm we use to compute the nearest neighbors is “brute”, and we specify “metric=cosine” 
        algorithm will calculate the cosine similarity between rating vectors. Finally, we fit the model.
        '''
        interest_user_rating = interest_user_rating.drop_duplicates(
            ['userID', 'bookTitle'])
        print(interest_user_rating)
        interest_user_rating_pivot = interest_user_rating.pivot(
            index='bookTitle', columns='userID', values='bookRating').fillna(0)
        print(interest_user_rating_pivot)
        # csr_matrix is sparse matrix
        interest_user_rating_matrix = csr_matrix(interest_user_rating_pivot.values)
        print(interest_user_rating_matrix)
        print(interest_user_rating)

        # implementation of KNN
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        model_knn.fit(interest_user_rating_matrix)
        print(model_knn)

        query_index = np.random.choice(interest_user_rating_pivot.shape[0])
        print(query_index)
        distances, indices = model_knn.kneighbors(
            interest_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1), n_neighbors=1)

        interest_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1)

        print(interest_user_rating_pivot)
        print("Recommendation for the book:-  ",
              interest_user_rating_pivot.index[query_index])

        for i in range(0, len(distances.flatten())):
            if i == 0:
                print('Recommendations for {0}:\n'.format(
                    interest_user_rating_pivot.index[query_index]))
            else:
                print('{0}: {1}, with distance of {2}'.format(
                    i, interest_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))


obj = recommendationClassKnn()
obj.show()
obj.shapeOfData()
obj.diagram()
obj.recommendation()
