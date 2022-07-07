import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from streamlit_assets.components import get_recos, display_anime


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
    return pd.read_csv('data/animes.csv').drop_duplicates(subset=['uid','title'])
@st.cache()    
def load_reviews():
    return pd.read_csv('data/reviews_light.csv', usecols=['uid','anime_uid','score'])

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
    with st.form('Anime'):
        anime_list  = st.multiselect('Select Animes you like',animes['title'].unique())
        submitted = st.form_submit_button('Gimme recommendations')
    if submitted:
        with st.spinner('**よし Yosh!**'):
            anime_ids = animes.loc[animes['title'].isin(anime_list), 'uid'].unique()
            df = get_recos(anime_ids,reviews)
            df_merged = animes.merge(df, on='uid').drop_duplicates(subset='uid').sort_values(by='match', ascending=False).reset_index(drop=True)
            st.session_state['df'] = df_merged
    
    if st.session_state['df'] is not None:
        sort_key = st.selectbox('Sort results by:',['match','score','popularity'])
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






