import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


class Books():
    
    def __init__(self):
        self.books = pd.read_csv('./Book/Books.csv')
        self.users = pd.read_csv('./Book/Users.csv')
        self.ratings = pd.read_csv('./Book/Ratings.csv')
        
        # Splitting Explicit and Implicit user ratings
        self.ratings_explicit = self.ratings[self.ratings.bookRating != 0]
        self.ratings_implicit = self.ratings[self.ratings.bookRating == 0]
        
        # Each Books Mean ratings and Total Rating Count
        self.average_rating = pd.DataFrame(self.ratings_explicit.groupby('ISBN')['bookRating'].mean())
        self.average_rating['ratingCount'] = pd.DataFrame(self.ratings_explicit.groupby('ISBN')['bookRating'].count())
        self.average_rating = self.average_rating.rename(columns = {'bookRating':'MeanRating'})
        
        # To get a stronger similarities
        counts1 = self.ratings_explicit['userID'].value_counts()
        self.ratings_explicit = self.ratings_explicit[self.ratings_explicit['userID'].isin(counts1[counts1 >= 50].index)]
        
        # Explicit Books and ISBN
        self.explicit_ISBN = self.ratings_explicit.ISBN.unique()
        self.explicit_books = self.books.loc[self.books['ISBN'].isin(self.explicit_ISBN)]
        
        # Look up dict for Book and BookID
        self.Book_lookup = dict(zip(self.explicit_books["ISBN"], self.explicit_books["bookTitle"]))
        self.ID_lookup = dict(zip(self.explicit_books["bookTitle"],self.explicit_books["ISBN"]))


        
    def Top_Books(self, n=10, RatingCount = 100, MeanRating = 3):
        
        BOOKS = self.books.merge(self.average_rating, how = 'right', on = 'ISBN')
        
        M_Rating = BOOKS.loc[BOOKS.ratingCount >= RatingCount].sort_values('MeanRating', ascending = False).head(n)

        H_Rating = BOOKS.loc[BOOKS.MeanRating >= MeanRating].sort_values('ratingCount', ascending = False).head(n)

            
        return M_Rating, H_Rating

    
class SVD(Books):
    
    def __init__(self, n_latent_factor = 50):
        super().__init__()
        self.n_latent_factor = n_latent_factor
        self.ratings_mat = self.ratings_explicit.pivot(index="userID", columns="ISBN", values="bookRating").fillna(0)
        
        self.uti_mat = self.ratings_mat.values
        # normalize by each users mean
        self.user_ratings_mean = np.mean(self.uti_mat, axis = 1)
        self.mat = self.uti_mat - self.user_ratings_mean.reshape(-1, 1)
        
        self.explicit_users = np.sort(self.ratings_explicit.userID.unique())
        self.User_lookup = dict(zip(range(1,len(self.explicit_users)),self.explicit_users))
        
        self.predictions = None

    def scipy_SVD(self):
        
        # singular value decomposition
        U, S, Vt = svds(self.mat, k = self.n_latent_factor)
        
        S_diag_matrix=np.diag(S)
        
        # Reconstructing Original Prediction Matrix
        X_pred = np.dot(np.dot(U, S_diag_matrix), Vt) + self.user_ratings_mean.reshape(-1, 1)
        
        self.predictions = pd.DataFrame(X_pred, columns = self.ratings_mat.columns, index = self.ratings_mat.index)
    
        return

    def Recommend_Books(self, userID, num_recommendations = 10):
        
        # Get and sort the user's predictions
        user_row_number = self.User_lookup[userID] # User ID starts at 1, not 0

        sorted_user_predictions = self.predictions.loc[user_row_number].sort_values(ascending=False) 
        
        # Get the user's data and merge in the books information.
        user_data = self.ratings_explicit[self.ratings_explicit.userID == (self.User_lookup[userID])]
        user_full = (user_data.merge(self.books, how = 'left', left_on = 'ISBN', right_on = 'ISBN').
                         sort_values(['bookRating'], ascending=False)
                     )
    
        # Recommend the highest predicted rating books that the user hasn't seen yet.
        recom = (self.books[~self.books['ISBN'].isin(user_full['ISBN'])].
                            merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
                                  left_on = 'ISBN',
                                  right_on = 'ISBN'))
        recom = recom.rename(columns = {user_row_number: 'Predictions'})
        recommend = recom.sort_values(by=['Predictions'], ascending = False)
        recommendations = recommend.iloc[:num_recommendations, :-1]
        
        return user_full, recommendations

        
class KNN(Books):

    def __init__(self, n_neighbors = 10):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.ratings_mat = self.ratings_explicit.pivot(index="ISBN", columns="userID", values="bookRating").fillna(0)
        self.uti_mat = csr_matrix(self.ratings_mat.values)
        
        # KNN Model Fitting
        self.model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
        self.model_knn.fit(self.uti_mat)
    

        
    def Recommend_Books(self, book, n_neighbors = 10):
            
        # Book Title  to BookID
        #bID = list(self.Book_lookup.keys())[list(self.Book_lookup.values()).index(book)]
        bID = self.ID_lookup[book]

        query_index = self.ratings_mat.index.get_loc(bID)
        
        KN = self.ratings_mat.iloc[query_index, :].values.reshape(1, -1)
                    
        distances, indices = self.model_knn.kneighbors(KN, n_neighbors = n_neighbors+1)
        
        Rec_books = list()
        Book_dis = list()
        
        for i in range(1, len(distances.flatten())):
            Rec_books.append(self.ratings_mat.index[indices.flatten()[i]])
            Book_dis.append(distances.flatten()[i])
        
        Book = self.Book_lookup[bID]
        
        Recommmended_Books = self.books[self.books['ISBN'].isin(Rec_books)]
        
        return Book, Recommmended_Books, Book_dis
            