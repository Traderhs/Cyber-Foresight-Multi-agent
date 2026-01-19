"""
Script for pulling historic data for Reddit API.

Documentation:
Reddit API:
https://praw.readthedocs.io/en/stable/index.html

Reddit Instance:
https://praw.readthedocs.io/en/stable/code_overview/reddit_instance.html

Subreddit class
https://praw.readthedocs.io/en/stable/code_overview/models/subreddit.html
"""
from typing import Dict, List
import praw
import pandas as pd
from datetime import datetime
import time

subreddits_list = [
    "arab",
    "arabasian",
    "asia",
    "asianpolitics",
    "asiareport",
    "askbalkans",
    "askhistorians",
    "asklatinamerica",
    "askreddit",
    "ausnews",
    "australiamedianews",
    "australianews",
    "australianpolitics",
    "breakingnews",
    "centralamerica",
    "channelnewsasia",
    "collapse",
    "combatfootage",
    "conspiracy",
    "credibledefense",
    "eastasianews",
    "economy",
    "economics",
    "eunews",
    "euro",
    "europe",
    "europeanforum",
    "europes",
    "foodforthought",
    "freeeuropenews",
    "geopolitics",
    "globalmarkets",
    "globalmarketnews",
    "globaltalk",
    "google",
    "history",
    "immigration",
    "internationalrelation",
    "inthenews",
    "middleeast",
    "middleeastnews",
    "military",
    "moderatepolitics",
    "news",
    "newsandpolitics",
    "news_world_conflicts",
    "newworldorder",
    "nuclearwar",
    "politicaldiscussion",
    "politics",
    "politicsall",
    "politicsandwar",
    "war",
    "worldnews",
    "worldpolitics",
    "worldwar"
]

countries_dict = {
    "US": "(USA OR America OR 'United States')",
    "GB": "(UK OR British OR 'United Kingdom' OR Britain)",
    "CA": "(Canada OR Canadian)",
    "AU": "(Australia)",
    "UA": "(Ukraine)",
    "RU": "(Russia)",
    "FR": "(France OR French)",
    "DE": "(German)",
    "BR": "(Brazil)",
    "CN": "(China OR Chinese)",
    "JP": "(Japan)",
    "PK": "(Pakistan)",
    "KP": "('North Korea')",
    "KR": "('South Korea')",
    "IN": "(India)",
    "TW": "(Taiwan)",
    "NL": "(NetherLands OR Holland OR Dutch)",
    "ES": "(Spain OR Spanish)",
    "SE": "(Sweden OR Swedish)",
    "MX": "(Mexic)",
    "IR": "(Iran)",
    "IL": "(Israel)",
    "SA": "(Saudi)",
    "SY": "(Syria)",
    "FI": "(Finland OR Finnish)",
    "IE": "(Ireland OR Irish)",
    "AT": "(Austria)",
    "NO": "(Norway OR Norwegian)",
    "CH": "(Switzerland OR Swiss)",
    "IT": "(Italy OR Italian)",
    "MY": "(Malaysia)",
    "EG": "(Egypt)",
    "TR": "(Turkey OR Turkish)",
    "PT": "(Portugal OR Portuguese)",
    "PS": "(Palestin OR 'West Bank' OR Gaza)",
    "AE": "(UAE OR 'United Arab Emirates' OR Emarat)"
}

# Date range
START_DATE = "2011-07"
END_DATE = "2025-05"


def aggr_dates(
    dates: Dict[str, List[datetime]],
    period_range: pd.PeriodIndex
) -> pd.DataFrame:
    """
    :param dates: Dictionary of the form {"Dates": [datetime, datetime, ...]}
    :param period_range: Range of dates in format YYYY-MM

    :return: Dataframe with counts of dates found per month, ordered by the
    preriod range PeriodIndex.
    """
    # Create mini dataframe and drop duplicate dates (probably same posts)
    count_df = pd.DataFrame(dates).drop_duplicates(subset="Dates")

    # Count of posts per month and re-indexing for all required months
    count_df = count_df["Dates"].dt.to_period("M").value_counts().sort_index()
    return count_df.reindex(period_range, fill_value=0)


if __name__ == "__main__":
    # Init output empty dataframe
    date_range = pd.period_range(START_DATE, END_DATE, freq="M")
    output_df = pd.DataFrame(index=date_range)
    output_df.index.name = "Date"

    # Init Reddit client
    reddit = praw.Reddit(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET",
        user_agent="PersonalScript by u/AcademicApp"
    )

    for code, country in countries_dict.items():
        col_name = f"War_Conflict_{code}"  # Column name with country data
        dates_dict = {"Dates": []}  # Dictionary to store posts dates

        # Search query
        query = f"({country} AND War) OR " + \
                f"({country} AND 'Military Conflict') OR " + \
                f"({country} AND 'Military Attack') OR " + \
                f"({country} AND 'Political Tension') OR " + \
                f"({country} AND 'Political Conflict') OR " + \
                f"({country} AND 'Armed Force Attack')"

        print(f"Searching posts for {country}...")

        for sub_reddit in subreddits_list:
            try:
                posts = reddit.subreddit(sub_reddit).search(
                    query,
                    sort="relevance",
                    time_filter="all",
                    limit=800
                )

                # Get date per post
                for post in posts:
                    post_date = datetime.utcfromtimestamp(post.created_utc)
                    dates_dict["Dates"].append(post_date)

                time.sleep(0.1)
            except Exception as e:
                print(f"Error fetching {code}: {e}")

        # Store country data to the output dataframe
        df = aggr_dates(dates_dict, date_range)
        output_df[col_name] = df.values

    # Create column with monthly sum across all countries.
    output_df["War_Conflict_All"] = output_df.sum(axis=1).astype(int)

    # Export as CSV
    output_csv = "RedditData.csv"
    output_df.to_csv(output_csv, index=True)
    print(f"Saved to {output_csv}")
