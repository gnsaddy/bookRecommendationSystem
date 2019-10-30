import BRS
import argparse
import sys
import pandas as pd

def parse_arguments():
    
    parser = argparse.ArgumentParser(description='SVD or KNN collaborative filtering')

    parser.add_argument(
        "--SVD",
        action="store_true",
        help="User based collaborative filtering using SVD"
    )
    parser.add_argument(
        "--KNN",
        action="store_true",
        help="Item collaborative filtering using KNN"
    )
    
    return parser.parse_args()

def YN():
    reply = str(input('\n\nContinue (y/n):\t')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return False


def main():
    
    args = parse_arguments()
    
    cont = True
    
    if not args.SVD and not args.KNN:
        print("\n\nChoose SVD or KNN for testing user/item based collaborative filtering\n")
        
        Top_B = BRS.Books()
        
        High_Mean_Rating, High_Rating_Count = Top_B.Top_Books()
        
        pd.set_option('display.max_colwidth', -1)
        
        print("\n\nBooks with high ratings :\n")
        print(High_Mean_Rating[['bookTitle','MeanRating','ratingCount','bookAuthor']])
        
        print("\n\nBooks with high rating count :\n")
        print(High_Rating_Count[['bookTitle','MeanRating','ratingCount','bookAuthor']])
        
        sys.exit()

    if args.SVD:
    
        UCF = BRS.SVD()
        
        UCF.scipy_SVD()
        
        while cont:
            
            try:
                User_ID = int(input('Enter User ID in the range {0}-{1}: '.format(1,len(UCF.explicit_users))))
            except:
                print('Enter a number')
                sys.exit()
            
            if User_ID in range(1,len(UCF.explicit_users)):
                pass
            else:
                print("Choose between {0}-{1}".format(1,len(UCF.explicit_users)))
                sys.exit()
            
            
            Rated_Books , SVD_Recommended_Books = UCF.Recommend_Books(userID=User_ID)
            
            pd.set_option('display.max_colwidth', -1)
            
            print("\nThe Books already  rated by the user\n")
            print(Rated_Books[['bookTitle','bookRating']])
            
            print("\nRecommended Books for the user\n")
            SVD_Recommended_Books = SVD_Recommended_Books.merge(UCF.average_rating, how='left', on='ISBN')
            SVD_Recommended_Books = SVD_Recommended_Books.rename(columns = {'bookRating':'MeanRating'})
            print(SVD_Recommended_Books[['bookTitle','MeanRating','bookAuthor']])
            
            cont = YN()
            
    if args.KNN:
        
        ICF = BRS.KNN()
        
        while cont:
            
            book_name = input('\n\nEnter the Book Title:\t')
    
            _, KNN_Recommended_Books, _ = ICF.Recommend_Books(book_name)
            
            print('Recommendations for the book --> {0}:\n'.format(book_name))
            
            KNN_Recommended_Books = KNN_Recommended_Books.merge(ICF.average_rating, how='left', on='ISBN')
            KNN_Recommended_Books = KNN_Recommended_Books.rename(columns = {'bookRating':'MeanRating'})
            
            print(KNN_Recommended_Books[['bookTitle','MeanRating','bookAuthor']])
            
            cont = YN()


if __name__ == '__main__':
    main()
