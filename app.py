import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Netflix Analytics Dashboard",
    layout="wide"
)

st.title("🎬 Netflix Data Analytics Dashboard")

st.markdown("Analyze Netflix Movies and TV Shows")

uploaded_file = st.file_uploader(
    "Upload Netflix Dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)

else:
    df = pd.read_excel("data/netflix_titles.xlsx")


df.fillna("Unknown", inplace=True)

if 'date_added' in df.columns:
    df['date_added'] = pd.to_datetime(
        df['date_added'],
        errors='coerce'
    )

    df['year_added'] = df['date_added'].dt.year

st.sidebar.header("Filters")

type_filter = st.sidebar.multiselect(
    "Select Type",
    df['type'].unique(),
    default=df['type'].unique()
)

rating_filter = st.sidebar.multiselect(
    "Select Rating",
    df['rating'].unique(),
    default=df['rating'].unique()
)

country_filter = st.sidebar.multiselect(
    "Select Country",
    sorted(df['country'].unique())[:50]
)

filtered_df = df[
    (df['type'].isin(type_filter)) &
    (df['rating'].isin(rating_filter))
]

if country_filter:
    filtered_df = filtered_df[
        filtered_df['country'].isin(country_filter)
    ]

total_titles = len(filtered_df)

movies_count = len(
    filtered_df[filtered_df['type'] == 'Movie']
)

tvshows_count = len(
    filtered_df[filtered_df['type'] == 'TV Show']
)

col1, col2, col3 = st.columns(3)

col1.metric("Total Titles", total_titles)
col2.metric("Movies", movies_count)
col3.metric("TV Shows", tvshows_count)

st.divider()

st.subheader("Movies vs TV Shows")

type_count = filtered_df['type'].value_counts().reset_index()

type_count.columns = ['Type', 'Count']

fig1 = px.pie(
    type_count,
    names='Type',
    values='Count',
    title='Distribution of Movies and TV Shows'
)

st.plotly_chart(fig1, use_container_width=True)

st.subheader("Content Added Over Years")

if 'year_added' in filtered_df.columns:

    yearly_content = (
        filtered_df['year_added']
        .value_counts()
        .sort_index()
        .reset_index()
    )

    yearly_content.columns = ['Year', 'Count']

    fig2 = px.line(
        yearly_content,
        x='Year',
        y='Count',
        markers=True,
        title='Netflix Content Trend'
    )

    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Top Genres")

genres = (
    filtered_df['listed_in']
    .str.split(', ')
    .explode()
)

genre_count = genres.value_counts().head(10).reset_index()

genre_count.columns = ['Genre', 'Count']

fig3 = px.bar(
    genre_count,
    x='Genre',
    y='Count',
    title='Top 10 Genres'
)

st.plotly_chart(fig3, use_container_width=True)

st.subheader("Top Countries")

countries = (
    filtered_df['country']
    .str.split(', ')
    .explode()
)

country_count = countries.value_counts().head(10).reset_index()

country_count.columns = ['Country', 'Count']

fig4 = px.bar(
    country_count,
    x='Country',
    y='Count',
    title='Top Countries Producing Netflix Content'
)

st.plotly_chart(fig4, use_container_width=True)

st.subheader("Rating Analysis")

rating_count = (
    filtered_df['rating']
    .value_counts()
    .reset_index()
)

rating_count.columns = ['Rating', 'Count']

fig5 = px.bar(
    rating_count,
    x='Rating',
    y='Count',
    title='Ratings Distribution'
)

st.plotly_chart(fig5, use_container_width=True)

st.subheader("Search Movies and TV Shows")

search = st.text_input("Enter Movie/Show Name")

if search:

    result = filtered_df[
        filtered_df['title']
        .str.contains(search, case=False)
    ]

    st.dataframe(
        result[['title', 'type', 'country', 'rating']]
    )


st.subheader("Recommendation System")

filtered_df['description'] = (
    filtered_df['description']
    .fillna('')
)

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(
    filtered_df['description']
)

cosine_sim = cosine_similarity(tfidf_matrix)

movie_titles = filtered_df['title'].values

selected_movie = st.selectbox(
    "Select Movie or TV Show",
    movie_titles
)

if st.button("Recommend"):

    idx = filtered_df[
        filtered_df['title'] == selected_movie
    ].index[0]

    similarity_scores = list(
        enumerate(cosine_sim[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:6]

    st.write("Recommended Titles:")

    for i in similarity_scores:

        movie_index = i[0]

        st.write(
            filtered_df.iloc[movie_index]['title']
        )
st.subheader("Dataset Preview")

st.dataframe(filtered_df.head())