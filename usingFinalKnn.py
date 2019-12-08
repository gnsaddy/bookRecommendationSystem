import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import argparse
import sys


class Books:

    def __init__(self):
        self.books = pd.read_csv('./Book/Books.csv')
        self.users = pd.read_csv('./Book/Users.csv')
        self.ratings = pd.read_csv('./Book/Ratings.csv')

        # Splitting Explicit and Implicit user ratings
        # we are removing the rating set which is having the rating as 0
        self.ratings_explicit = self.ratings[self.ratings.bookRating != 0]
        self.ratings_implicit = self.ratings[self.ratings.bookRating == 0]

        # Each Books Mean ratings and Total Rating Count
        self.average_rating = pd.DataFrame(
            self.ratings_explicit.groupby('ISBN')['bookRating'].mean())
        self.average_rating['ratingCount'] = pd.DataFrame(
            self.ratings_explicit.groupby('ISBN')['bookRating'].count())
        self.average_rating = self.average_rating.rename(
            columns={'bookRating': 'MeanRating'})

        # To get a stronger similarities
        counts1 = self.ratings_explicit['userID'].value_counts()
        self.ratings_explicit = self.ratings_explicit[
            self.ratings_explicit['userID'].isin(counts1[counts1 >= 50].index)]

        # Explicit Books and ISBN
        self.explicit_ISBN = self.ratings_explicit.ISBN.unique()
        self.explicit_books = self.books.loc[self.books['ISBN'].isin(
            self.explicit_ISBN)]

        # Look up dict for Book and BookID
        self.Book_lookup = dict(
            zip(self.explicit_books["ISBN"], self.explicit_books["bookTitle"]))
        self.ID_lookup = dict(
            zip(self.explicit_books["bookTitle"], self.explicit_books["ISBN"]))

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

    def Top_Books(self, n=10, RatingCount=100, MeanRating=3):
        # here we are specifying the latency of meanRating with value of 3
        # and latency of RatingCount with value of 100
        # this makes a threshold value for predicting the best possible book sets for the user
        # books with the highest rating
        # this function will not recommend any books just shows the highest rated books rated by every user
        BOOKS = self.books.merge(self.average_rating, how='right', on='ISBN')
        # print(Books)
        M_Rating = BOOKS.loc[BOOKS.ratingCount >= RatingCount].sort_values(
            'MeanRating', ascending=False).head(n)

        H_Rating = BOOKS.loc[BOOKS.MeanRating >= MeanRating].sort_values(
            'ratingCount', ascending=False).head(n)

        # print(M_Rating)
        # print(H_Rating)

        return M_Rating, H_Rating


class KNN(Books):

    def __init__(self, n_neighbors=5):
        # calling super class __init__ method
        super().__init__()
        # assigning k  value = 5
        self.n_neighbors = n_neighbors
        # removing nan value
        self.ratings_mat = self.ratings_explicit.pivot(
            index="ISBN", columns="userID", values="bookRating").fillna(0)
        '''
        Implementing kNN
        In numerical analysis and scientific computing, a sparse matrix or sparse array is a matrix in which
        most of the elements are zero.
        We convert our table to a 2D matrix, and fill the missing values with zeros 
        (since we will calculate distances between rating vectors). We then transform the values(ratings)
        of the matrix dataframe into a scipy sparse matrix for more efficient calculations.
        Finding the Nearest Neighbors We use unsupervised algorithms with sklearn.neighbors.
        The algorithm we use to compute the nearest neighbors is “brute”, and we specify “metric=cosine” 
        algorithm will calculate the cosine similarity between rating vectors. Finally, we fit the model.
        '''
        self.uti_mat = csr_matrix(self.ratings_mat.values)
        # KNN Model Fitting
        # KNN Model Fitting
        # using cosine similarity
        '''Mathematically, it measures 
        the cosine of the angle between two vectors projected in a multi-dimensional space
        Cosine similarity is a metric used to determine how
         similar the documents are irrespective of their size.'''
        self.model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model_knn.fit(self.uti_mat)

    def Recommend_Books(self, book, n_neighbors=10):
        # Book Title  to BookID
        # bID = list(self.Book_lookup.keys())[list(self.Book_lookup.values()).index(book)]
        bID = self.ID_lookup[book]

        query_index = self.ratings_mat.index.get_loc(bID)

        KN = self.ratings_mat.iloc[query_index, :].values.reshape(1, -1)

        distances, indices = self.model_knn.kneighbors(
            KN, n_neighbors=n_neighbors + 1)

        Rec_books = list()
        Book_dis = list()

        for i in range(1, len(distances.flatten())):
            Rec_books.append(self.ratings_mat.index[indices.flatten()[i]])
            Book_dis.append(distances.flatten()[i])

        Book = self.Book_lookup[bID]

        Recommmended_Books = self.books[self.books['ISBN'].isin(Rec_books)]

        return Book, Recommmended_Books, Book_dis


def YN():
    reply = str(input('\n\nContinue (y/n):\t')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return False


def MainCall():
    cont = True

    while cont:
        print("\n1-Top Books\n2-Recommendation based on Book title\n3-Exit\n")
        choice = int(input("Enter choice :- "))
# ---------------------------------------------------------------------------------------------------------------------------------------
        if choice == 1:
            print("\n\nKNN item based collaborative filtering\n")

            Top_B = Books()
            Books().diagram()

            High_Mean_Rating, High_Rating_Count = Top_B.Top_Books()

            pd.set_option('display.max_colwidth', -1)

            print("\n\nBooks having highest ratings :\n")
            print(
                High_Mean_Rating[['bookTitle', 'MeanRating', 'ratingCount', 'bookAuthor']])
            print("\n\nBooks having highest rating count :\n")
            print(High_Rating_Count[['bookTitle',
                                     'MeanRating', 'ratingCount', 'bookAuthor']])
            print("\nFor getting recommendation based on Knn pass --KNN as argument ")
# --------------------------------------------------------------------------------------------------------------------------------
        if choice == 2:
            ICF = KNN()
            # Books().diagram()

            while cont:
                book_name = input('\n\nEnter the Book Title:\t')

                try:
                    _, KNN_Recommended_Books, _ = ICF.Recommend_Books(
                        book_name)
                    print(
                        'Recommendations for the book --> {0}:\n'.format(book_name))

                    KNN_Recommended_Books = KNN_Recommended_Books.merge(
                        ICF.average_rating, how='left', on='ISBN')
                    KNN_Recommended_Books = KNN_Recommended_Books.rename(
                        columns={'bookRating': 'MeanRating'})

                    print(KNN_Recommended_Books[[
                        'bookTitle', 'MeanRating', 'bookAuthor']])
                except KeyError:
                    print("Book title not found.\t Try for another book title")

                # for research book title
                cont = YN()
# ---------------------------------------------------------------------------------------------------------------
        if choice == 3:
            print("Thank you for using BRS")
            sys.exit()
# ----------------------------------------------------------------------------------------------------------------

        # for choice search
        cont = YN()


MainCall()
