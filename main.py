from flask import Flask, render_template, request

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity




rating_popular_movie=pd.read_csv('recom.csv')
S=set(rating_popular_movie.title)
L=list(S)

movie_features_df=rating_popular_movie.pivot_table(index=['title'],columns='userId',values='rating').fillna(0)
movie_features_df['id']=[i for i in range(882)]
from scipy.sparse import csr_matrix

movie_features_df_matrix = csr_matrix(movie_features_df.values)

from sklearn.neighbors import NearestNeighbors


model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(movie_features_df_matrix)

def get_index_from_title(title):
  if title in S:
      return movie_features_df[movie_features_df.index == title]["id"].values[0]
  
  return "no"

def get_recommendations(movie_user_likes):
     L=[]
     movie_index = get_index_from_title(movie_user_likes)
     if movie_index=='no':
       return 0
     else:
        distances, indices = model_knn.kneighbors(movie_features_df.iloc[movie_index,:].values.reshape(1, -1), n_neighbors = 6)
        for i in range(1, len(distances.flatten())):
            L.append(movie_features_df.index[indices.flatten()[i]])
        return(L)



app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')
    
@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = get_recommendations(movie)
    if r==0:
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')



if __name__ == '__main__':
    app.run()




