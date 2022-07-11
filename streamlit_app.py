import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from streamlit_assets.components import get_recos, display_anime, test_algo
from surprise import SVD, NMF, KNNBasic, SVDpp
import plotly.express as px


# PAGE SETUP
st.set_page_config(
    page_title="reco system",
    layout="wide",
    page_icon="streamlit_assets/assets/icon_logo.PNG",

)

# From https://discuss.streamlit.io/t/how-to-center-images-latex-header-title-etc/1946/4
with open("streamlit_assets/style.css") as f:
    st.markdown("""<link href='http://fonts.googleapis.com/css?family=Roboto:400,100,100italic,300,300italic,400italic,500,500italic,700,700italic,900italic,900' rel='stylesheet' type='text/css'>""", unsafe_allow_html=True)
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


left_column, center_column,right_column = st.columns([1,4,1])

with left_column:
    st.info("Project using streamlit")
with right_column:
    st.write("##### Authors\nThis tool has been developed by [Emile D. Esmaili](https://github.com/emileDesmaili)")
with center_column:
    st.image("streamlit_assets/assets/app_logo.PNG",width=600)

# data import
@st.cache()
def load_animes():
    df = pd.read_csv('data_local/animes.csv').drop_duplicates(subset=['uid','title']).reset_index()
    df = df.rename(columns={"uid": "item_id"})
    return df
@st.cache()    
def load_reviews():
    df = pd.read_csv('data_local/reviews_light.csv', usecols=['uid','anime_uid','score'])
    df.columns=['user_id','item_id','rating']
    return df

animes = load_animes()
reviews = load_reviews()

# Menu 

st.sidebar.write(f'# Welcome')

page_container = st.sidebar.container()
with page_container:
    page = option_menu("Menu", ["Recommender", 'About'], 
    icons=['reddit','info'], menu_icon="cast", default_index=0)

#Reco

if page =='Recommender':
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    # FORM
    with st.form('Anime'):
        anime_list  = st.multiselect('Select Animes you like',animes['title'].unique())
        slider = st.slider('How much of the recommendation is driven by similar user preferences vs genres (0 is only genre, 1 is only users)',0.,1.,step=0.05, value=0.5)
        submitted = st.form_submit_button('Gimme recommendations 🚀')
    if (submitted and len(anime_list) == 0):
        st.warning('Please add at least one anime')

    # COMPUTATION
    elif submitted:
        with st.spinner('✨ **よし Yosh!** ✨'):
            anime_ids = animes.loc[animes['title'].isin(anime_list), 'item_id'].unique()
            df = get_recos(anime_ids,animes, reviews, slider).head(12)
            st.session_state['df'] = df
            
    # DISPLAY
    if st.session_state['df'] is not None:
        sort_key = st.selectbox('Sort results by:',['YourMatch','user match','genre match','score','popularity'])
        st.session_state['df'] = st.session_state['df'].sort_values(by=sort_key, ascending=False).reset_index(drop=True)
        #st.write(st.session_state['df'])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            display_anime(st.session_state['df'].iloc[0])
        with col2:
            display_anime(st.session_state['df'].iloc[1])
        with col3:
            display_anime(st.session_state['df'].iloc[2])
        with col4:
            display_anime(st.session_state['df'].iloc[3])
        col5, col6, col7, col8  = st.columns(4)
        with col5:
            display_anime(st.session_state['df'].iloc[4])
        with col6:
            display_anime(st.session_state['df'].iloc[5])
        with col7:
            display_anime(st.session_state['df'].iloc[6])
        with col8:
            display_anime(st.session_state['df'].iloc[7])
        col9, col10, col11, col12 = st.columns(4)
        with col9:
            display_anime(st.session_state['df'].iloc[8])
        with col10:
            display_anime(st.session_state['df'].iloc[9])
        with col11:
            display_anime(st.session_state['df'].iloc[10])
        with col12:
            display_anime(st.session_state['df'].iloc[11])

if page == 'About':
    st.subheader('Data & Assumptions')
    st.markdown(f'- User-based Recommender Systems build using **[Surprise](https://surprise.readthedocs.io/en/stable/index.html#)**', unsafe_allow_html=True)
    st.markdown(f'- Genre-based Recommender Systems built using a co-occurence matrix of genres & averaged on all items', unsafe_allow_html=True)
    st.markdown(f'- I used this **[dataset](https://www.kaggle.com/datasets/marlesson/myanimelist-dataset-animes-profiles-reviews) from Kaggle** which I would like to update ', unsafe_allow_html=True)
    st.write('- When the user adds animes, instead of asking them to score each one, **I assign all of them with a rating of 8.0/10**. This is a big assumption and stands to be improved')
    
    st.subheader('Algorithms')
    st.write('I used SVD as a baseline, it performed better than NMF and on par with KNN which is slower')
    model_names = ['SVD','NMF','SVD++','K-NN']
    models = [SVD(),NMF(), SVDpp(),KNNBasic()]
    model_dict = dict(zip(model_names,models))
    if 'table' not in st.session_state:
        st.session_state['table'] = None

    with st.form('Training Parameters'):
        col1, col2 = st.columns(2)
        with col1:
            algo = st.selectbox('Select Algorithm',model_names)
            algo_selection = model_dict[algo]
        with col2:
            k = st.slider('K-fold Cross-validation',2,10,step=1)
            measures = st.multiselect('Evaluation Metrics',['RMSE', 'MAE','MSE'],['RMSE', 'MAE'])
        submitted = st.form_submit_button('Train Algorithm')
    if submitted:
        with st.spinner('Training Algorithm...'):
            table = test_algo(reviews, algo_selection, k, measures)
            st.session_state['table'] = table
    
    if st.session_state['table'] is not None:
        col1, col2 = st.columns(2)
        with col1:
            df = st.session_state['table']
            droplist = ['fit_time','test_time']
            fig = px.bar(df, x= df.index, y = df.drop(droplist,axis=1).columns, barmode='group')
            st.plotly_chart(fig)
        with col2:
            st.write(st.session_state['table'])
    









