"""Point-in-time inputs from published GPR releases and recent GDELT headlines."""

from io import BytesIO
import json
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree
from pathlib import Path
import re
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

import pandas as pd

GPR_ARCHIVE = "https://www.matteoiacoviello.com/gpr_archive_files"
GPR_PAGE = "https://www.matteoiacoviello.com/gpr.htm"
GPR_SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "processed" / "gpr_releases.csv"
GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"
NEWS_QUERY = {
    "United States": '("United States" OR "Wall Street" OR "Federal Reserve")',
    "Euro area": '("Eurozone" OR "European Union" OR "European Central Bank")',
    "Spain": '("Spain" OR "España" OR "IBEX")',
}
TOPICS = '(sanctions OR tariffs OR geopolitical OR election OR oil)'


def read_url(url, timeout=8):
    with urlopen(Request(url, headers={"User-Agent": "ESG Market Volatility Research Dashboard"}),
                 timeout=timeout) as response:
        return response.read(6_000_000)


def summarize_release(data, release_date):
    recent = pd.read_stata(data)
    recent["date"] = pd.to_datetime(recent["DAY"].astype(str), format="%Y%m%d")
    recent = recent.sort_values("date").tail(7)
    if recent.empty or recent["date"].max() > release_date:
        raise ValueError("Invalid GPR observation dates for release.")
    columns = {"GPRD": "gpr", "GPRD_ACT": "acts", "GPRD_THREAT": "threats"}
    values = recent[list(columns)].rename(columns=columns).mean().to_dict()
    # Only use the release starting the following day because the hour is unknown.
    return {"available_date": release_date + pd.Timedelta(days=1),
            "observed_through": recent["date"].max(), **values}


def load_press_context(refresh=True):
    archive = pd.read_csv(GPR_SNAPSHOT, parse_dates=["available_date", "observed_through"])
    if not refresh:
        return archive, False
    try:
        page = read_url(GPR_PAGE).decode("utf-8", errors="replace")
        releases = re.findall(r"Last update:\s*(?:<[^>]+>\s*)*([A-Za-z]+ \d{1,2},\s+\d{4})", page)
        release = pd.Timestamp(" ".join(releases[1].split()))
        if release + pd.Timedelta(days=1) <= archive["available_date"].max():
            return archive, True
        url = f"{GPR_ARCHIVE}/data_gpr_daily_recent_{release:%Y%m%d}.dta"
        row = summarize_release(BytesIO(read_url(url)), release)
        return pd.concat([archive, pd.DataFrame([row])], ignore_index=True), True
    except (OSError, ValueError, IndexError, KeyError):
        return archive, False


def gdelt_headlines(region):
    query = f"{NEWS_QUERY.get(region, TOPICS)} {TOPICS}" if region in NEWS_QUERY else TOPICS
    params = urlencode({"query": query, "mode": "ArtList", "format": "json",
                        "timespan": "7d", "maxrecords": 25, "sort": "DateDesc"})
    payload = json.loads(read_url(f"{GDELT_API}?{params}"))
    articles = []
    seen = set()
    for item in payload.get("articles", []):
        title, url = item.get("title", "").strip(), item.get("url", "")
        key = " ".join(re.findall(r"\w+", title.casefold()))
        when = pd.to_datetime(item.get("seendate"), format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
        if not key or key in seen or pd.isna(when):
            continue
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            continue
        seen.add(key)
        articles.append({"Headline": title, "Source": parsed.hostname, "Seen (UTC)": when, "Article": url})
    return sorted(articles, key=lambda article: article["Seen (UTC)"], reverse=True)[:10]


NEWS_LOCALES = {
    "United States": ("en-US", "US", "US:en", "US market"),
    "Euro area": ("en-GB", "GB", "GB:en", "European markets"),
    "Spain": ("es", "ES", "ES:es", "IBEX bolsa"),
}


def recent_headlines(region):
    try:
        articles = gdelt_headlines(region)
        if articles:
            return articles, "GDELT", "Seen (UTC)"
    except (OSError, ValueError, KeyError):
        pass

    language, country, edition, market = NEWS_LOCALES[region]
    query = urlencode({"q": f"{market} (geopolitics OR tariffs OR aranceles OR petróleo) when:7d",
                       "hl": language, "gl": country, "ceid": edition})
    try:
        feed = ElementTree.fromstring(read_url(f"https://news.google.com/rss/search?{query}"))
    except ElementTree.ParseError as exc:
        raise ValueError("Invalid news RSS response.") from exc
    articles = []
    seen = set()
    for item in feed.findall("./channel/item"):
        title, url = item.findtext("title", ""), item.findtext("link", "")
        key = " ".join(re.findall(r"\w+", title.casefold()))
        if not key or key in seen or urlsplit(url).scheme != "https":
            continue
        try:
            when = pd.Timestamp(parsedate_to_datetime(item.findtext("pubDate", ""))).tz_convert("UTC")
        except (ValueError, TypeError):
            continue
        seen.add(key)
        articles.append({"Headline": title, "Source": item.findtext("source", ""),
                         "Published (UTC)": when, "Article": url})
    return sorted(articles, key=lambda article: article["Published (UTC)"], reverse=True)[:10], "Google News", "Published (UTC)"
