"""
Script for pulling historic data for Google searches of keywords related to
war conflict and political tension from Google Trends API. The API has an
unknown rate limit, which is easily reached and blockes requests after a point.

Moreover, no data are returned for North Korea ("KP"). For time windows longer
than 5 years, data are returned at monthly frequency with the first day of
each month as a timestamp.
"""
from itertools import product
import time
import pandas as pd
from pytrends.request import TrendReq

# 36 countries, no data for KP (North Korea Republic)
countries = [
    "US", "GB", "CA", "AU", "UA", "RU", "FR", "DE", "BR", "CN", "JP", "PK",
    "KP", "KR", "IN", "TW", "NL", "ES", "SE", "MX", "IR", "IL", "SA", "SY",
    "FI", "IE", "AT", "NO", "CH", "IT", "MY", "EG", "TR", "PT", "PS", "AE"
]

keywords = ["war"]

# "war conflict",
# "military attack",
# "armed force attack",
# "political conflict"

FILE = "trends_war.csv"
TIME_WINDOW = "2011-07-01 2025-03-31"


def rename_df(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Rename the data returned from a single search to camel-case, adding a
    suffix with an underscore and the country code, e.g., "WarConflict_US".
    Additionally, "isPartial" column is dropped.

    :param df: DataFrame returned from a query to Google Trends API.
    :param country: 2-letter country code e.g., "US".

    :return: The ipdated dataframe.
    """
    df.drop(columns=["isPartial"], inplace=True)

    # Get the data column name and break it down with space as separator.
    col_name = df.columns[0]
    col_name_parts = col_name.split(" ")

    if len(col_name_parts) > 1:

        # Logic for column break down with more than 1 part
        col_name_parts = [part.capitalize() for part in col_name_parts]
        col_name = f"{''.join(col_name_parts)}_{country}"
        df.columns = [col_name]
        return df

    # Logic for column break down with only 1 part
    df.columns = [f"{col_name.capitalize()}_{country}"]
    return df


if __name__ == "__main__":
    # Init session and empty output dataframe
    trends_requester = TrendReq(hl="en-US", tz=0)
    output_df = pd.DataFrame()

    for code, keyword in product(countries, keywords):
        try:
            print(f"Code: {code}", f"| Keyword: {keyword}")
            trends_requester.build_payload(
                kw_list=[keyword],
                timeframe=TIME_WINDOW,
                geo=code
            )

            # Get result as dataframe (datetime index, keyword, isPartial)
            result_df = trends_requester.interest_over_time()
            result_df = rename_df(result_df, code)

            # Logic if output dataframe already has data
            if len(output_df) > 0:
                main_column = result_df.columns[0]
                output_df[main_column] = result_df[main_column].values

            # If this is the first query, init output as a df copy
            else:
                output_df = result_df.copy()
        except Exception as e:
            print(f"Failure for {keyword} in {code} - {e}")

        # Sleep to avoid blocking API access from excessive requests
        time.sleep(10)

    output_df.index.name = "Date"
    output_df.to_csv(FILE, index=True)
