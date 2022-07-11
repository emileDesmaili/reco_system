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
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp



def filter_reviews(reviews, min_item_ratings=500, min_user_ratings= 1):
    """function to filter out items with few reviews

    Args:
        reviews (_type_): dataframe of reviews with columns ['user_id','item_id','rating']
        min_item_ratings (int, optional): minimum ratings filter. Defaults to 500.
        min_user_ratings (int, optional): minimum reviews per user filter. Defaults to 1.

    Returns:
        dataframe: filtered dataframe
    """
    filter_items = reviews['item_id'].value_counts() > min_item_ratings
    filter_items = filter_items[filter_items].index.tolist()

    filter_users = reviews['user_id'].value_counts() > min_user_ratings
    filter_users = filter_users[filter_users].index.tolist()
    df = reviews[(reviews['item_id'].isin(filter_items)) & (reviews['user_id'].isin(filter_users))]
    return df

def get_top_n(predictions, uid, n=12):
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

    # Then sort the predictions for each user and retrieve the n highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:]

        return top_n

def get_recos_users(new_ids, reviews, filter_items=True, filter_n=500, default_rating=8):
    """generate recommendations from a list of items with matrix factorization

    Args:
        anime_ids (list): list of item ids
        reviews (dataframe): dataframe of reviews with columns ['user_id','item_id','rating']
        default_rating (integer or float): default rating assigned to all new items

    Returns:
        dataframe: dataframe of recommendations for a user with item_id and score
    """
    # renaming columns just in case
    reviews.columns = ['user_id','item_id','rating']
    if filter_items == True:
        df = filter_reviews(reviews,min_item_ratings=filter_n)
    else:
        df = reviews
    # add anime list to dataframe 
    new_uid = max(df['user_id'])+1
    

    #adding user reviews to dataset
    new_list = list(zip([new_uid]*len(new_ids),list(new_ids),list(default_rating*np.ones(len(new_ids)))))
    new_df = pd.DataFrame(new_list, columns=['user_id','item_id','rating'])
    df_reviews = df.append(new_df).reset_index(drop=True)
    df_reviews['rating'] = df_reviews['rating'].apply(float)
    df_reviews = df_reviews.dropna()
    # preprocess
    reader = Reader(rating_scale=(0, max(df_reviews['rating'])))
    data = Dataset.load_from_df(df_reviews[['user_id', 'item_id', 'rating']], reader)
    # Retrieve the trainset.
    trainset = data.build_full_trainset()
    # Build an algorithm, and train it.
    algo = SVD()
    algo.fit(trainset)

    # Then predict ratings for all pairs (u, i) that are NOT in the training set.
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    top_n = get_top_n(predictions,new_uid, n=12)

    # create clean dataframe with item_ids and scores
    ids = []
    scores = []
    for item in top_n[new_uid]:
        ids.append(item[0])
        scores.append(item[1])
        df = pd.DataFrame(list(zip(list(ids),scores)),columns=['item_id','user match']) 
    return df

def get_recos_genre(new_ids, animes):
    
    #preprocessing
    genres = animes["genre"].apply(ast.literal_eval).apply(lambda x: ', '.join(x))
    idx = animes[animes['item_id'].isin(new_ids)].index
    anime_ids = list(animes.iloc[idx]['item_id'].values)
    
    # count vectorizer and occurence matrix
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(genres)
    Xc = (X[idx,:]* X.T).toarray()
    df = pd.DataFrame(Xc.T, index=animes['item_id'],columns=anime_ids)

    #removing users input & normalizing
    mask = df.index.isin(anime_ids)
    df_masked = df[~mask]
    df_norm = df_masked / df_masked.max()
    #averaging all scores (not great but can't think of better yet)
    df_norm['genre match'] = df_norm.mean(axis=1)
    reco_genre = df_norm
    return reco_genre

def get_recos(new_ids, animes, reviews, slider):

    df = get_recos_users(new_ids,reviews)
    df2 = get_recos_genre(new_ids,animes)
    st.write(df)
    st.write(df2)
    df_reco = df.merge(df2,on='item_id')
    df_reco['YourMatch'] = df_reco['user match']*slider + (1-slider)*10*df_reco['genre match']
    df_merged = animes.merge(df_reco, on='item_id').drop_duplicates(subset='item_id').sort_values(by='YourMatch', ascending=False).reset_index(drop=True)
    st.write(df_merged)
    return df_merged

def display_anime(df):
    title = df['title']
    score = df['score']
    reco_score = int(10*df['YourMatch'])
    user_score = int(10*df['user match'])
    genre_score = int(100*df['genre match'])
    link = df['link']
    img = df['img_url']
    synopsis = df['synopsis']
    dates = df['aired']
    genres = ast.literal_eval(df['genre'])


    st.markdown(f'#### [{title}]({link})')
    st.write(f"Match: **{reco_score}%**  - (user: {user_score}% - genre: {genre_score}%)")
    st.image(img)
    
    st.write(f"User Rating: **{score}**")
    
    st.multiselect('Tags',options=genres,default=genres, key=title)

    with st.expander('Details'):
        st.write(f"Aired: {dates}")
        st.write(synopsis)
        

def test_algo(reviews, algo, k, measures):
    df = filter_reviews(reviews)
    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
    # Run k-fold cross-validation and print results.
    return pd.DataFrame(cross_validate(algo, data, measures=measures, cv=k, verbose=True))










        






