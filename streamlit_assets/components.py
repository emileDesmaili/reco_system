import streamlit as st
import pandas as pd 
import itertools
import numpy as np
from surprise import SVD, NMF, KNNBasic, SVDpp
from surprise.model_selection import cross_validate
from surprise import Dataset
from surprise import Reader
from collections import defaultdict
import ast






def get_recos(anime_ids, reviews):

    def get_top_n(predictions, n=12):
        """Return the top-N recommendation for each user from a set of predictions.
        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.
        Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        """

        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n

    #filter anime with few reviews
    min_anime_ratings = 500
    filter_anime = reviews['anime_uid'].value_counts() > min_anime_ratings
    filter_anime = filter_anime[filter_anime].index.tolist()

    min_user_ratings = 1
    filter_users = reviews['uid'].value_counts() > min_user_ratings
    filter_users = filter_users[filter_users].index.tolist()

    df = reviews[(reviews['anime_uid'].isin(filter_anime)) & (reviews['uid'].isin(filter_users))]
    # add anime list to dataframe 
    new_uid = max(df['uid'])+1
    new_list = list(zip([new_uid]*len(anime_ids),list(anime_ids),list(8*np.ones(len(anime_ids)))))
    new_df = pd.DataFrame(new_list, columns=['uid','anime_uid','score'])
    df_reviews = df.append(new_df).reset_index(drop=True)
    df_reviews['score'] = df_reviews['score'].apply(int)
    # preprocess
    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(df_reviews[['uid', 'anime_uid', 'score']], reader)
    # Retrieve the trainset.
    trainset = data.build_full_trainset()
    # Build an algorithm, and train it.
    algo = SVD()
    algo.fit(trainset)

    # Then predict ratings for all pairs (u, i) that are NOT in the training set.
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)

    top_n = get_top_n(predictions, n=12)

    ids = []
    scores = []
    for item in top_n[new_uid]:
        ids.append(item[0])
        scores.append(item[1])
        df = pd.DataFrame(list(zip(list(ids),scores)),columns=['uid','match'])
        
    
    return df

def display_anime(df):
    title = df['title']
    match = int(10*df['match'])
    score = df['score']
    link = df['link']
    img = df['img_url']
    synopsis = df['synopsis']
    dates = df['aired']
    genres = ast.literal_eval(df['genre'])

    st.markdown(f'#### [{title}]({link})')
    st.write(f"Match: **{match}%**")
    st.image(img)
    
    st.write(f"User Rating: **{score}**")
    
    st.multiselect('Tags',options=genres,default=genres)

    with st.expander('Details'):
        st.write(f"Aired: {dates}")
        st.write(synopsis)
        

def test_algo(reviews, algo, k, measures):
    min_anime_ratings = 500
    filter_anime = reviews['anime_uid'].value_counts() > min_anime_ratings
    filter_anime = filter_anime[filter_anime].index.tolist()

    min_user_ratings = 1
    filter_users = reviews['uid'].value_counts() > min_user_ratings
    filter_users = filter_users[filter_users].index.tolist()
    df = reviews[(reviews['anime_uid'].isin(filter_anime)) & (reviews['uid'].isin(filter_users))]

    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(df[['uid', 'anime_uid', 'score']], reader)
    
    # Run k-fold cross-validation and print results.
    return pd.DataFrame(cross_validate(algo, data, measures=measures, cv=k, verbose=True))









        






