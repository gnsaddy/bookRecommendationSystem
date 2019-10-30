# library-book-recommendation-system
 
# Book-Recommendation---Collaborative-Filtering
Book Recommendation System using SVD and KNN for User/Item based collaborative filtering

## Running the Book Recommendation

The program recommends books for a particular User based on CF using singular-value decomposition (SVD) algorithm `--SVD` and recommends books related to a particular book based on CF using k-Nearest Neighbors algorithm `--KNN`. Both the algorithms are run on explicit user ratings.

### Example

For recommendations for a particular user (by default SVD latent factor is set to 50). 
Enter the user ID within the displayed range when prompted. 
The output prints the books already rated by the user and then the top 10 recommendations.

```
python main.py --SVD
```

For recommendations for a particular Book (by default K-Neighbors is set to 10). 
Enter the book title when prompted. (Choose Book Title from the `./Book/Books(explicit).csv` file)
The output prints the Top 10 recommended books.

 ```
 python main.py --KNN
 ```
 
 If no arguements were passed, the top 10 books with mean highest ratings and top 10 books with the high ratings count.
 
 ```
 python main.py
 ```

