
import asyncio
from concurrent.futures import ProcessPoolExecutor
from io import StringIO

import aiohttp
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


st.set_page_config(page_title="Temperature Analysis", layout="wide")

month_to_season = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn"
}


def current_season_from_timestamp(ts: pd.Timestamp) -> str:
    return month_to_season[ts.month]

def add_rolling_features(city_df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    city_df = city_df.sort_values("timestamp").copy()
    city_df["rolling_mean_30"] = city_df["temperature"].rolling(window=window, min_periods=1).mean()
    city_df["rolling_std_30"] = city_df["temperature"].rolling(window=window, min_periods=1).std().fillna(0.0)
    return city_df


def seasonal_statistics(city_df: pd.DataFrame) -> pd.DataFrame:
    stat = (
        city_df.groupby(["city", "season"], as_index=False)["temperature"]
        .agg(
            season_mean="mean",
            season_std="std",
            season_min="min",
            season_max="max",
            observations="count"
        )
    )
    stat["lower_bound"] = stat["season_mean"] - 2 * stat["season_std"]
    stat["upper_bound"] = stat["season_mean"] + 2 * stat["season_std"]
    return stat


def detect_anomalies(city_df: pd.DataFrame, stat_df: pd.DataFrame) -> pd.DataFrame:
    merged = city_df.merge(
        stat_df[["city", "season", "season_mean", "season_std", "lower_bound", "upper_bound"]],
        on=["city", "season"],
        how="left",
    )

    merged["rolling_lower_bound"] = merged["rolling_mean_30"] - 2 * merged["rolling_std_30"]
    merged["rolling_upper_bound"] = merged["rolling_mean_30"] + 2 * merged["rolling_std_30"]

    merged["is_anomaly"] = (
        (merged["temperature"] < merged["rolling_lower_bound"]) |
        (merged["temperature"] > merged["rolling_upper_bound"])
    )
    return merged


def add_linear_trend(city_df: pd.DataFrame) -> pd.DataFrame:
    city_df = city_df.sort_values("timestamp").copy()
    x = np.arange(len(city_df))
    slope, intercept = np.polyfit(x, city_df["temperature"].to_numpy(), 1)
    city_df["trend"] = intercept + slope * x
    city_df["trend_slope"] = slope
    return city_df


def analyze(city_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    city_df = add_rolling_features(city_df)
    stat_df = seasonal_statistics(city_df)
    city_df = detect_anomalies(city_df, stat_df)
    city_df = add_linear_trend(city_df)
    return city_df, stat_df

def proc_for_par(city_df: pd.DataFrame):
    return analyze(city_df)

@st.cache_data
def analyze_aeq(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    city_frames = []
    stat_frames = []

    for _, city_df in data.groupby("city", sort=True):
        analyzed_city, city_stat = analyze(city_df.copy())
        city_frames.append(analyzed_city)
        stat_frames.append(city_stat)

    return pd.concat(city_frames, ignore_index=True), pd.concat(stat_frames, ignore_index=True)

@st.cache_data
def analyze_par(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = [city_df.copy() for _, city_df in data.groupby("city", sort=True)]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(proc_for_par, grouped))

    city_frames = [item[0] for item in results]
    stat_frames = [item[1] for item in results]
    return pd.concat(city_frames, ignore_index=True), pd.concat(stat_frames, ignore_index=True)


CITY_TO_QUERY = {
    "New York": "New York",
    "London": "London",
    "Paris": "Paris",
    "Tokyo": "Tokyo",
    "Moscow": "Moscow",
    "Sydney": "Sydney",
    "Berlin": "Berlin",
    "Beijing": "Beijing",
    "Rio de Janeiro": "Rio de Janeiro",
    "Dubai": "Dubai",
    "Los Angeles": "Los Angeles",
    "Singapore": "Singapore",
    "Mumbai": "Mumbai",
    "Cairo": "Cairo",
    "Mexico City": "Mexico City",
}

def get_cur_temp_sync(city: str, api_key: str) -> dict:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": CITY_TO_QUERY[city], "appid": api_key, "units": "metric"}
    response = requests.get(url, params=params, timeout=20)
    data = response.json()

    if response.status_code == 401:
        raise ValueError(data)
    if response.status_code != 200:
        raise RuntimeError(data)

    return {
        "city": city,
        "temperature": data["main"]["temp"],
        "weather": data["weather"][0]["description"],
        "raw": data,
    }

async def get_cur_temp_async(city: str, api_key: str) -> dict:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": CITY_TO_QUERY[city], "appid": api_key, "units": "metric"}
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, params=params) as response:
            data = await response.json()

            if response.status == 401:
                raise ValueError(data)
            if response.status != 200:
                raise RuntimeError(data)

            return {
                "city": city,
                "temperature": data["main"]["temp"],
                "weather": data["weather"][0]["description"],
                "raw": data,
            }

def check_temp(city: str, current_temp: float, stat_df: pd.DataFrame, season: str) -> dict:
    row = stat_df[(stat_df["city"] == city) & (stat_df["season"] == season)].iloc[0]
    is_normal = row["lower_bound"] <= current_temp <= row["upper_bound"]

    return {
        "city": city,
        "season": season,
        "current_temperature": current_temp,
        "historical_mean": row["season_mean"],
        "historical_std": row["season_std"],
        "lower_bound": row["lower_bound"],
        "upper_bound": row["upper_bound"],
        "is_normal": bool(is_normal),
    }

def build_time_series_figure(city_data: pd.DataFrame, city: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=city_data["timestamp"],
        y=city_data["temperature"],
        mode="lines",
        name="temperature",
    ))
    fig.add_trace(go.Scatter(
        x=city_data["timestamp"],
        y=city_data["rolling_mean_30"],
        mode="lines",
        name="rolling_mean_30",
    ))
    fig.add_trace(go.Scatter(
        x=city_data["timestamp"],
        y=city_data["trend"],
        mode="lines",
        name="trend",
    ))

    anomalies = city_data[city_data["is_anomaly"]]
    fig.add_trace(go.Scatter(
        x=anomalies["timestamp"],
        y=anomalies["temperature"],
        mode="markers",
        name="anomalies",
    ))

    fig.update_layout(title=f"Time series for {city}", xaxis_title="Date", yaxis_title="Temperature, °C")
    return fig

def season_profile_figure(city_stat: pd.DataFrame, city: str):
    fig = px.bar(
        city_stat.sort_values("season"),
        x="season",
        y="season_mean",
        error_y="season_std",
        title=f"Seasonal profile for {city}",
    )
    return fig

def load_data(upl_file):
    if upl_file is None:
        df = pd.read_csv("temperature_data.csv")
    else:
        df = pd.read_csv(upl_file)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "season" not in df.columns:
        df["season"] = df["timestamp"].dt.month.map(month_to_season)

    return df

st.title("Анализ температурных данных")
st.sidebar.header("Параметры")
upl_file = st.sidebar.file_uploader("Загрузите CSV с историческими данными", type=["csv"])
api_key = st.sidebar.text_input("OpenWeatherMap API key", type="password")
sel_md = st.sidebar.radio("Режим анализа", ["sequential", "parallel"], index=0)
try:
    df = load_data(upl_file)
except Exception as exc:
    st.error(f"Не удалось загрузить данные: {exc}")
    st.stop()
cities = sorted(df["city"].dropna().unique().tolist())
selected_city = st.sidebar.selectbox("Выберите город", cities)
if sel_md == "sequential":
    analyzed_df, stat_df = analyze_aeq(df)
else:
    analyzed_df, stat_df = analyze_par(df)

city_data = analyzed_df[analyzed_df["city"] == selected_city].sort_values("timestamp")
city_stat = stat_df[stat_df["city"] == selected_city].sort_values("season")

st.subheader("Описательная статистика")
st.dataframe(city_data[["temperature", "rolling_mean_30", "rolling_std_30"]].describe())
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Количество наблюдений", len(city_data))
with col2:
    st.metric("Количество аномалий", int(city_data["is_anomaly"].sum()))
with col3:
    st.metric("Доля аномалий", f'{city_data["is_anomaly"].mean():.2%}')

st.subheader("Временной ряд и аномалии")
st.plotly_chart(build_time_series_figure(city_data, selected_city), use_container_width=True)
st.subheader("Сезонный профиль")
st.plotly_chart(season_profile_figure(city_stat, selected_city), use_container_width=True)
st.subheader("Таблица сезонной статистики")
st.dataframe(city_stat)
st.subheader("Текущая температура через OpenWeatherMap")
if not api_key.strip():
    st.info("API key не введён. Данные текущей погоды не показываются.")
else:
    try:
        weather_sync = get_cur_temp_sync(selected_city, api_key.strip())
        current_temp = weather_sync["temperature"]
        now_season = current_season_from_timestamp(pd.Timestamp.utcnow())
        result = check_temp(selected_city, current_temp, stat_df, now_season)

        st.write(f'Текущая температура в {selected_city}: **{current_temp:.2f} °C**')
        st.write(f'Описание погоды: **{weather_sync["weather"]}**')
        st.write(f'Текущий сезон: **{now_season}**')
        st.write(
            f'Историческая норма: от **{result["lower_bound"]:.2f}** до **{result["upper_bound"]:.2f} °C** '
            f'(mean={result["historical_mean"]:.2f}, std={result["historical_std"]:.2f})'
        )

        if result["is_normal"]:
            st.success("Текущая температура находится в пределах исторической нормы.")
        else:
            st.error("Текущая температура является аномальной относительно исторических данных.")

        with st.expander("Проверка асинхронного запроса"):
            async_result = asyncio.run(get_cur_temp_async(selected_city, api_key.strip()))
            st.json({
                "city": async_result["city"],
                "temperature": async_result["temperature"],
                "weather": async_result["weather"],
            })
            st.caption(
                "Для одного города синхронный запрос проще. "
                "Асинхронный удобен при множестве одновременных запросов."
            )

    except ValueError as exc:
        error_data = exc.args[0] if exc.args else {}
        if isinstance(error_data, dict) and error_data.get("cod") == 401:
            st.error(
                f'Ошибка API OpenWeatherMap: cod={error_data.get("cod")}, '
                f'message="{error_data.get("message")}"'
            )
        else:
            st.error(f"Некорректный ответ API: {error_data}")

    except Exception as exc:
        st.error(f"Ошибка при запросе текущей погоды: {exc}")
