import usingFinalKnn
import argparse
import sys
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(description='KNN collaborative filtering')

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

    if not args.KNN:
        print("\n\nKNN item based collaborative filtering\n")

        Top_B = usingFinalKnn.Books()
        usingFinalKnn.Books().diagram()

        High_Mean_Rating, High_Rating_Count = Top_B.Top_Books()

        pd.set_option('display.max_colwidth', -1)

        print("\n\nBooks hiving highest ratings :\n")
        print(High_Mean_Rating[['bookTitle', 'MeanRating', 'ratingCount', 'bookAuthor']])
        print("\n\nBooks having highest rating count :\n")
        print(High_Rating_Count[['bookTitle', 'MeanRating', 'ratingCount', 'bookAuthor']])
        print("\nFor getting recommendation based on Knn pass --KNN as argument ")
        sys.exit()

    if args.KNN:

        ICF = usingFinalKnn.KNN()

        while cont:
            book_name = input('\n\nEnter the Book Title:\t')

            _, KNN_Recommended_Books, _ = ICF.Recommend_Books(book_name)

            print('Recommendations for the book --> {0}:\n'.format(book_name))

            KNN_Recommended_Books = KNN_Recommended_Books.merge(ICF.average_rating, how='left', on='ISBN')
            KNN_Recommended_Books = KNN_Recommended_Books.rename(columns={'bookRating': 'MeanRating'})

            print(KNN_Recommended_Books[['bookTitle', 'MeanRating', 'bookAuthor']])

            cont = YN()


if __name__ == '__main__':
    main()
