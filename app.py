import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import io
import os

# TMDb API settings
def _load_api_key() -> str:
    """Read the TMDb key from Streamlit secrets or the TMDB_API_KEY env var."""
    try:
        if "TMDB_API_KEY" in st.secrets:
            return st.secrets["TMDB_API_KEY"]
    except Exception:
        pass
    return os.environ.get("TMDB_API_KEY", "")

TMDB_API_KEY = _load_api_key()
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/tv"
TMDB_SERIES_DETAILS_URL = "https://api.themoviedb.org/3/tv/{series_id}"
TMDB_EPISODES_URL = "https://api.themoviedb.org/3/tv/{series_id}/season/{season_num}"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"


# Page configuration
st.set_page_config(page_title="TV Series Ratings Heatmap", layout="wide")
st.title("📺 TV Series Ratings Heatmap")

if not TMDB_API_KEY:
    st.error(
        "No TMDb API key found. Add `TMDB_API_KEY` to `.streamlit/secrets.toml` "
        "or set it as an environment variable. See the README."
    )
    st.stop()

# TMDb search helper
@st.cache_data(show_spinner=False)
def search_tmdb(query: str):
    params = {"api_key": TMDB_API_KEY, "query": query, "language": "en-US", "include_adult": False}
    res = requests.get(TMDB_SEARCH_URL, params=params)
    return res.json().get("results", []) if res.status_code == 200 else []

# TMDb details helper
@st.cache_data(show_spinner=False)
def get_details(series_id: int):
    url = TMDB_SERIES_DETAILS_URL.format(series_id=series_id)
    res = requests.get(url, params={"api_key": TMDB_API_KEY, "language": "en-US"})
    return res.json() if res.status_code == 200 else {}

# Episode ratings helper
@st.cache_data(show_spinner=False)
def fetch_episode_ratings(series_id: int, n_seasons: int):
    ratings = []
    for season_num in range(1, n_seasons + 1):
        url = TMDB_EPISODES_URL.format(series_id=series_id, season_num=season_num)
        res = requests.get(url, params={"api_key": TMDB_API_KEY, "language": "en-US"})
        episodes = res.json().get("episodes", []) if res.status_code == 200 else []
        ratings.append([ep.get("vote_average", np.nan) for ep in episodes])
    return ratings

# Sidebar input
st.sidebar.header("🔍 Search for Series")
query = st.sidebar.text_input("Series name:")
if st.sidebar.button("Search"):
    if not query.strip():
        st.sidebar.error("Please enter a series name.")
        st.stop()
    # Keep results across reruns so picking another series from the list works
    st.session_state["results"] = search_tmdb(query)

if "results" in st.session_state:
    results = st.session_state["results"]
    if not results:
        st.sidebar.error("No series found. Try another name.")
        st.stop()

    # Display search results
    titles = [f"{r['name']} ({r.get('first_air_date','')[:4]})" for r in results]
    selection = st.sidebar.selectbox("Select series:", titles)
    idx = titles.index(selection)
    serie = results[idx]

    # Fetch series details
    info = get_details(serie['id'])
    title = info.get('name', 'Unknown')
    poster_path = info.get('poster_path')
    poster_url = f"{TMDB_IMG_BASE}{poster_path}" if poster_path else None
    overview = info.get('overview', '')
    avg_rating = info.get('vote_average')
    num_seasons = info.get('number_of_seasons', 1)

    # Build ratings matrix
    all_ratings = fetch_episode_ratings(serie['id'], num_seasons)
    max_eps = max((len(s) for s in all_ratings), default=0)
    data = np.full((max_eps, num_seasons), np.nan)
    for si, season in enumerate(all_ratings):
        for ei, val in enumerate(season):
            data[ei, si] = val

    # Prepare DataFrame
    rows = [f"E{i+1}" for i in range(max_eps)]
    cols = [f"S{s+1}" for s in range(num_seasons)]
    df = pd.DataFrame(data, index=rows, columns=cols)

    # Plot heatmap with Matplotlib (square cells)
    colors = ['#d73027','#fc8d59','#fee08b','#d9ef8b','#91cf60','#1a9850']
    cmap = ListedColormap(colors)
    bounds = [0,4,5.5,7,8,9,10]
    norm = BoundaryNorm(bounds, cmap.N)
    # Determine cell size
    cell_size = 0.5
    fig, ax = plt.subplots(
        figsize=(cell_size * num_seasons, cell_size * max_eps)
    )
    im = ax.imshow(df.values, cmap=cmap, norm=norm)
    ax.set_aspect('equal')  # make cells square
    # Annotate
    for (j, i), v in np.ndenumerate(df.values):
        if not np.isnan(v):
            ax.text(i, j, f"{v:.1f}", ha='center', va='center', fontsize=6)
    # Labels
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel('Season')
    ax.set_ylabel('Episode')
    ax.set_title(f"Rating Heatmap: {title}")
    plt.tight_layout()

    # Layout
    left, right = st.columns([1,3])
    with left:
        if poster_url:
            st.image(poster_url, caption=title, use_container_width=True)
        st.markdown(f"### {title}")
        if avg_rating is not None:
            st.markdown(f"**TMDb Avg:** {avg_rating:.1f}")
        with st.expander("Overview", expanded=False):
            st.write(overview)
        with st.expander("Legend & Bands", expanded=False):
            legend_text = (
                "- <4: Very Bad  \n"
                "- 4–5.5: Bad  \n"
                "- 5.5–7: So-So  \n"
                "- 7–8: Good  \n"
                "- 8–9: Great  \n"
                "- ≥9: Masterpiece"
            )
            st.markdown(legend_text)
    with right:
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        st.download_button(
            "Download Heatmap as PNG",
            buf,
            file_name=f"{title}_heatmap.png",
            mime="image/png"
        )
else:
    st.info("Enter a series name and click Search in the sidebar.")
