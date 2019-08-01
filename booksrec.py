# Dependencies
from flask import Flask, request, jsonify
from flask import render_template
from sklearn.externals import joblib
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy import sparse
import traceback
import pandas as pd
import numpy as np
import flask
import regex as re

book_info_unique = pd.read_csv('./datasets/book_info_unique.csv')
book_pivot_10books = pd.read_csv('./datasets/book_pivot_10books.csv')
book_df = pd.read_csv('./datasets/book_df.csv')
cosine_sim = np.load('./datasets/cosine_sim.npy', allow_pickle=True)
dict_a = np.load('./datasets/inddict.npy', allow_pickle=True).item()
book_info = pd.read_csv('./datasets/book_info.csv')
popular_df = pd.read_csv('./datasets/popular_df_10.csv')
book_to_rate = pd.read_csv('./datasets/book_to_rate.csv')
pic = [str(i)+".jpg" for i in range(1,11)]
book_info.genre = book_info.genre.apply(lambda x: re.sub('[^A-Za-z0-9,]','', x).split(","))

# Your API definition
app = Flask(__name__)


@app.route('/')
@app.route('/index/')
def index():
    return render_template('index.html', popular_df=popular_df, book_to_rate=book_to_rate)


@app.route('/index/choose_book/', methods=['POST', 'GET'])
@app.route('/choose_book/', methods=['POST', 'GET'])
def choose():
    return render_template('choose_book.html', book_dict=dict_a)


@app.route('/index/choose_book/content_based', methods=['POST'])
@app.route('/choose_book/content_based', methods=['POST'])
@app.route('/content_based', methods=['POST'])
def content_rec():
    # retrieve the chosen book_id from the autocomplete menu on the index page
    # choose the matrix here and sent the df over to content_based
    book_id = request.form['book_title']
    # if no book was chosen, return to index page
    if book_id == '':
        return render_template('index.html', book_dict=dict_a)
   
    else:

        book_id = int(book_id)
        book_rec = content_recommender(book_id)

    return render_template('content_based.html', book_info=book_info,book_id=book_id, book_df=book_rec)


@app.route('/index/rate_book', methods=['POST', 'GET'])
@app.route('/rate_book', methods=['POST', 'GET'])
def rate_book():

    return render_template('rate_book.html', book_to_rate=book_to_rate)


@app.route('/index/collab', methods=['POST','GET'])
@app.route('/collab', methods=['POST','GET'])
def collab_rec():
    # retrieve the chosen book_id from the autocomplete menu on the index page
    book_rating = request.form.to_dict()
    book_rating = list(book_rating.values())
    book_df_c = collab_recommender(book_rating)
    return render_template('collab.html', book_df=book_df_c)


def content_recommender(book_id):
    similarity_scores = list(enumerate(cosine_sim[book_id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:31]

    #Get the books index
    books_index = [i[0] for i in similarity_scores]
    book_info_content = book_info.iloc[books_index]
    #Return the top 10 most similar books using integar-location based indexing (iloc)
    #Keep only 2 books with the same author
    return book_info_content[book_info_content.book_id.isin(book_info_unique['book_id'])].head(10)

def collab_recommender(rating_list):
    # get user input

    book_to_rate_id = [3,4,5,7,8,13,14,16,20,23]
    user_rating = [int(i) for i in rating_list] # change data type to int

    new_df = pd.DataFrame.from_dict({'book_id': book_to_rate_id,'rating': user_rating})
    new_df['user_id'] = 'new_user' # give a random user_id for new user
    new_df = new_df.pivot(index='user_id', columns='book_id', values='rating')
    new_df = csr_matrix(new_df.values)
    book_pivot_index = book_pivot_10books.set_index('user_id')
    book_pivot_10books_sparse = csr_matrix(book_pivot_index.values)
    # find similar user with the new user
    cosine_sim = cosine_similarity(book_pivot_10books_sparse, new_df)
    # form a dataframe using cosine_sim
    cosine_df = pd.DataFrame(cosine_sim, index=book_pivot_index.index, columns=['cosine'])
    similar_user = cosine_df.sort_values(by='cosine', ascending=False).reset_index().loc[0,'user_id']

    # recommend based on the similar user's preference
    user = int(similar_user/5000)
    cf_preds_df =  pd.read_pickle('./datasets/cf_preds_df'+str(user)+'.pkl')
    sorted_df = cf_preds_df[similar_user].sort_values(ascending=False)
    recommend_df = sorted_df[~sorted_df.index.isin(book_to_rate['book_id'])]
    del cf_preds_df
    recommend_df_unique = recommend_df[recommend_df.index.isin(book_info_unique.book_id)]#

    return book_info[book_info['book_id'].isin(recommend_df_unique.head(10).index)]

@app.route('/background/')
def background():
    return render_template('background.html')

@app.route('/EDA/')
def EDA():
    return render_template('EDA.html')

@app.route('/RS/')
def RS():
    return render_template('RS.html')

@app.route('/conclusion/')
def conclusion():
    return render_template('conclusion.html')

def convert(input):
    return re.sub('[^A-Za-z0-9,]','',input).split(",")

#operation that converts a template into a complete HTML page is called rendering.

#----- MAIN SENTINEL -----#
if __name__ == '__main__':
    '''Connects to the server'''

#     HOST = '127.0.0.1'
#     PORT = '5000'

    app.run(debug=True)
