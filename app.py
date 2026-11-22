
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="VELORA | Netflix Analytics",
    page_icon="🎬",
    layout="wide"
)

CSV_PATH = "netflix_titles.csv"

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #101010;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #181818;
}

h1, h2, h3 {
    color: #E50914 !important;
}

div[data-testid="stMetric"] {
    background-color: #1c1c1c;
    border: 1px solid #333333;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #aaaaaa;
}

div[data-testid="stMetricValue"] {
    color: white;
}

.stButton button {
    background-color: #E50914;
    color: white;
    border-radius: 8px;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(CSV_PATH)

    df.fillna("", inplace=True)

    df["release_year"] = pd.to_numeric(
        df["release_year"], errors="coerce"
    )

    return df

data = load_data()

# --------------------------------------------------
# PREPARATION
# --------------------------------------------------

data["type"] = data["type"].astype(str)
data["rating"] = data["rating"].astype(str)
data["listed_in"] = data["listed_in"].astype(str)
data["country"] = data["country"].astype(str)

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🎬 VELORA")

st.sidebar.caption("Netflix Content Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Explore Content",
        "Search & Filters",
        "Recommendations"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Netflix Content Analysis & Recommendation System")

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.title("VELORA")
    st.subheader("Netflix Content Overview")

    st.markdown(
        "Explore Netflix's content catalogue through data and visual insights."
    )

    total_titles = len(data)

    total_movies = len(
        data[data["type"] == "Movie"]
    )

    total_shows = len(
        data[data["type"] == "TV Show"]
    )

    total_countries = data["country"].str.split(",").explode().str.strip().nunique()

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Titles", f"{total_titles:,}")
    col2.metric("Movies", f"{total_movies:,}")
    col3.metric("TV Shows", f"{total_shows:,}")
    col4.metric("Countries", f"{total_countries:,}")

    st.markdown("---")

    # CONTENT DISTRIBUTION

    st.subheader("Content Distribution")

    type_counts = data["type"].value_counts().reset_index()
    type_counts.columns = ["Content Type", "Count"]

    fig = px.pie(
        type_counts,
        names="Content Type",
        values="Count",
        hole=0.55,
        color_discrete_sequence=["#E50914", "#B20710"]
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101010",
        plot_bgcolor="#101010"
    )

    st.plotly_chart(fig, use_container_width=True)

    # RELEASE YEAR TREND

    st.subheader("Content Released Over the Years")

    year_data = data.dropna(subset=["release_year"])

    year_counts = (
        year_data.groupby("release_year")
        .size()
        .reset_index(name="Count")
    )

    fig = px.area(
        year_counts,
        x="release_year",
        y="Count",
        color_discrete_sequence=["#E50914"]
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101010",
        plot_bgcolor="#101010",
        xaxis_title="Release Year",
        yaxis_title="Number of Titles"
    )

    st.plotly_chart(fig, use_container_width=True)

    # RECENT ADDITIONS

    st.subheader("Recently Added Titles")

    if "date_added" in data.columns:

        recent = data.copy()

        recent["date_added_parsed"] = pd.to_datetime(
            recent["date_added"],
            errors="coerce"
        )

        recent = recent.sort_values(
            "date_added_parsed",
            ascending=False
        )

        st.dataframe(
            recent[
                ["title", "type", "date_added", "release_year", "rating"]
            ].head(10),
            use_container_width=True,
            hide_index=True
        )

# --------------------------------------------------
# EXPLORE CONTENT
# --------------------------------------------------

elif page == "Explore Content":

    st.title("Explore Netflix Content")

    st.write(
        "Browse the catalogue and discover movies and TV shows."
    )

    # CONTENT TYPE FILTER

    content_type = st.selectbox(
        "Content Type",
        ["All", "Movie", "TV Show"]
    )

    explore_data = data.copy()

    if content_type != "All":

        explore_data = explore_data[
            explore_data["type"] == content_type
        ]

    # RANDOM DISCOVERY

    st.subheader("Discover Something")

    if st.button("🎲 Surprise Me"):

        if not explore_data.empty:

            random_title = explore_data.sample(1).iloc[0]

            st.success(f"Your pick: {random_title['title']}")

            st.write("**Type:**", random_title["type"])
            st.write("**Release Year:**", random_title["release_year"])
            st.write("**Rating:**", random_title["rating"])
            st.write("**Genre:**", random_title["listed_in"])
            st.write("**Description:**", random_title["description"])

    # CONTENT TABLE

    st.subheader("Catalogue")

    st.write(f"Showing {len(explore_data):,} titles")

    st.dataframe(
        explore_data[
            [
                "title",
                "type",
                "release_year",
                "rating",
                "duration",
                "listed_in",
                "country"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# SEARCH AND FILTERS
# --------------------------------------------------

elif page == "Search & Filters":

    st.title("Search & Filter Netflix Titles")

    st.write(
        "Find titles based on your preferred content type, "
        "rating, genre and release year."
    )

    filtered_data = data.copy()

    # TITLE SEARCH

    search_query = st.text_input(
        "🔍 Search by Title",
        placeholder="Enter movie or TV show name..."
    )

    if search_query:

        filtered_data = filtered_data[
            filtered_data["title"].str.contains(
                search_query,
                case=False,
                na=False
            )
        ]

    # CONTENT TYPE

    types = ["All"] + sorted(
        data["type"].dropna().unique().tolist()
    )

    selected_type = st.selectbox(
        "Content Type",
        types
    )

    if selected_type != "All":

        filtered_data = filtered_data[
            filtered_data["type"] == selected_type
        ]

    # RATINGS

    ratings = ["All"] + sorted(
        [r for r in data["rating"].unique() if r != ""]
    )

    selected_rating = st.selectbox(
        "Rating",
        ratings
    )

    if selected_rating != "All":

        filtered_data = filtered_data[
            filtered_data["rating"] == selected_rating
        ]

    # GENRE

    all_genres = sorted(
        set(
            genre.strip()
            for genres in data["listed_in"]
            for genre in genres.split(",")
            if genre.strip()
        )
    )

    selected_genre = st.selectbox(
        "Genre",
        ["All"] + all_genres
    )

    if selected_genre != "All":

        filtered_data = filtered_data[
            filtered_data["listed_in"].str.contains(
                selected_genre,
                case=False,
                na=False,
                regex=False
            )
        ]

    # RELEASE YEAR

    years = data["release_year"].dropna()

    if not years.empty:

        min_year = int(years.min())
        max_year = int(years.max())

        selected_years = st.slider(
            "Release Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )

        filtered_data = filtered_data[
            filtered_data["release_year"].between(
                selected_years[0],
                selected_years[1]
            )
        ]

    # RESULTS

    st.markdown("---")

    st.subheader("Search Results")

    st.metric("Matching Titles", len(filtered_data))

    st.dataframe(
        filtered_data[
            [
                "title",
                "type",
                "release_year",
                "rating",
                "duration",
                "listed_in",
                "description"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# RECOMMENDATION SYSTEM
# --------------------------------------------------

elif page == "Recommendations":

    st.title("VELORA Recommendation System")

    st.write(
        "Discover titles similar to your favourites using "
        "content-based filtering and machine learning."
    )

    # COMBINE MULTIPLE CONTENT FEATURES

    @st.cache_data
    def prepare_recommendation_model(df):

        features = (
            df["title"].astype(str) + " " +
            df["director"].astype(str) + " " +
            df["cast"].astype(str) + " " +
            df["listed_in"].astype(str) + " " +
            df["description"].astype(str)
        )

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        tfidf_matrix = vectorizer.fit_transform(features)

        return vectorizer, tfidf_matrix

    vectorizer, tfidf_matrix = prepare_recommendation_model(data)

    # TITLE SELECTION

    titles = data["title"].dropna().tolist()

    selected_title = st.selectbox(
        "Select a movie or TV show",
        titles
    )

    number_recommendations = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=10,
        value=5
    )

    if st.button("✨ Get Recommendations"):

        selected_index = data[
            data["title"] == selected_title
        ].index[0]

        # COSINE SIMILARITY

        similarity_scores = cosine_similarity(
            tfidf_matrix[selected_index],
            tfidf_matrix
        ).flatten()

        similar_indices = similarity_scores.argsort()[::-1]

        # REMOVE THE SELECTED TITLE

        similar_indices = [
            i for i in similar_indices
            if i != selected_index
        ]

        recommended_indices = similar_indices[
            :number_recommendations
        ]

        recommendations = data.iloc[
            recommended_indices
        ].copy()

        recommendations["Similarity Score"] = (
            similarity_scores[recommended_indices] * 100
        ).round(2)

        st.success(
            f"Recommendations similar to **{selected_title}**"
        )

        # DISPLAY RESULTS

        for _, row in recommendations.iterrows():

            with st.container(border=True):

                st.subheader(row["title"])

                col1, col2, col3 = st.columns(3)

                col1.write(f"🎬 **Type:** {row['type']}")
                col2.write(f"📅 **Year:** {row['release_year']}")
                col3.write(f"⭐ **Rating:** {row['rating']}")

                st.write(f"**Genre:** {row['listed_in']}")

                st.write(
                    f"**Similarity:** {row['Similarity Score']}%"
                )

                if row["description"]:

                    st.write(row["description"])

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    "<center style='color:gray;'>"
    "VELORA | Netflix Content Analytics & Recommendation System"
    "</center>",
    unsafe_allow_html=True
)
