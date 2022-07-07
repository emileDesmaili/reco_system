import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from streamlit_assets.components import get_recos


# PAGE SETUP
st.set_page_config(
    page_title="reco system",
    layout="wide",
    page_icon="streamlit_assets/assets/p4k_logo.png",

)

# From https://discuss.streamlit.io/t/how-to-center-images-latex-header-title-etc/1946/4
with open("streamlit_assets/style.css") as f:
    st.markdown("""<link href='http://fonts.googleapis.com/css?family=Roboto:400,100,100italic,300,300italic,400italic,500,500italic,700,700italic,900italic,900' rel='stylesheet' type='text/css'>""", unsafe_allow_html=True)
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


left_column, center_column,right_column = st.columns([1,3,1])

with left_column:
    st.info("Project using streamlit")
with right_column:
    st.write("##### Authors\nThis tool has been developed by [Emile D. Esmaili](https://github.com/emileDesmaili)")
with center_column:
    st.image("streamlit_assets/assets/app_logo.PNG")

# data import
@st.cache()
def load_animes():
    return pd.read_csv('data/animes.csv')
@st.cache()    
def load_reviews():
    return pd.read_csv('data/reviews.csv', usecols=['uid','anime_uid','score'])

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
    with st.form('Anime'):
        anime_list  = st.multiselect('Select Animes you like',animes['title'].unique())
        submitted = st.form_submit_button('Gimme recommendations')
    if submitted:
        with st.spinner('...'):
            anime_ids = animes.loc[animes['title'].isin(anime_list), 'uid'].unique()
            df = get_recos(anime_ids,reviews)
        st.write(df)



