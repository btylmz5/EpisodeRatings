# 📺 Episode Ratings Heatmap

A small Streamlit app that shows a TV series' episode ratings as a season-by-episode heatmap, so you can see at a glance where a show peaks and where it falls off.

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white) ![TMDb](https://img.shields.io/badge/data-TMDb-01B4E4)

## Features

- **Search any series** through the TMDb API and pick the right one from the results
- **Heatmap** with seasons as columns and episodes as rows, each cell showing its rating
- **Color bands** from *Very Bad* (< 4) to *Masterpiece* (≥ 9)
- **Series info** next to the chart: poster, overview, and TMDb average
- **Download** the heatmap as a PNG

## Getting Started

### 1. Get a TMDb API key

Create a free account at [themoviedb.org](https://www.themoviedb.org/) and request an API key under *Settings → API*.

### 2. Install

```bash
git clone https://github.com/btylmz5/EpisodeRatings.git
cd EpisodeRatings
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your key

Copy the example secrets file and put your key in it:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

```toml
# .streamlit/secrets.toml
TMDB_API_KEY = "your-tmdb-api-key"
```

You can also set it as an environment variable instead: `export TMDB_API_KEY=...`. `secrets.toml` is in `.gitignore`, so your key won't be committed.

### 4. Run

```bash
streamlit run app.py
```

## Rating Bands

| Rating | Band |
|---|---|
| < 4 | Very Bad |
| 4 – 5.5 | Bad |
| 5.5 – 7 | So-So |
| 7 – 8 | Good |
| 8 – 9 | Great |
| ≥ 9 | Masterpiece |

## Built With

Streamlit · pandas · NumPy · Matplotlib · [TMDb API](https://developer.themoviedb.org/)

> This product uses the TMDb API but is not endorsed or certified by TMDb.
