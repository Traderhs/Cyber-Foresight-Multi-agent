"""
Created on Jan 21 01:20:38 2023

@author: Zaid Almahmoud

This script counts for each country the number of tweets about wars and
conflicts related to that country. The output is the monthly count of these
tweets for each country in the period between July 2011 and December 2022.

Output file: ACA.csv
"""

# For sending GET requests from the API
import requests

# For saving access tokens & for file management
# when creating & adding to the dataset
import os

# To add wait time between requests
import time
from csv import writer
from typing import Dict, Tuple

# Place your Twitter API bearer token here!
os.environ['TOKEN'] = ''


def create_url(
    search_str: str,
    start_date: str,
    end_date: str,
    max_results: int = 10
) -> Tuple[str, dict]:
    """
    Create query data (URL, parameters) for Twitter API.

    :param search_str: API query string.
    :param start_date: String for starting timestamp (beginning of the month).
    :param end_date: String for ending timestamp (end of the month).
    :param max_results: Max number of returned results.

    :return search_url: API endpoint URL to query.
    :return query_params: Dictionary with query parameters.
    """
    # Change to the endpoint you want to collect data from
    search_url = "https://api.twitter.com/2/tweets/search/all"

    # Change params based on the endpoint you are using
    query_params = {
        'query': search_str,
        'start_time': start_date,
        'end_time': end_date,
        'max_results': max_results,
        'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
        'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
        'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
        'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
        'next_token': {}
    }

    return search_url, query_params


def auth() -> str:
    """Get the Twitter API bearer token value."""
    return os.getenv('TOKEN')


def create_headers(token: str) -> Dict[str, str]:
    """
    Create dictionary with API bearer token for Twitter API access.

    :param token: The API token.
    :return header: The dictionary for API access.
    """
    headers = {'Authorization': 'Bearer {}'.format(token)}
    return headers


def connect_to_endpoint(
    url: str,
    headers: Dict[str, str],
    params: dict,
    next_token: str = None
) -> dict:
    """
    Connect to API and query the endpoint.

    :param url: API endpoint URL to query.
    :param headers: Header dictionary for API querying.
    :param params: Dictionary with API query parameters.
    :param next_token: The API token.

    :return: Query response in json format.
    """
    # params object received from create_url function
    params['next_token'] = next_token
    response = requests.request('GET', url, headers=headers, params=params)
    print('Endpoint Response Code: ' + str(response.status_code))

    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def is_leap_year(year: int) -> bool:
    """Determine whether a year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


months = [
    '01', '02', '03', '04', '05', '06',
    '07', '08', '09', '10', '11', '12'
]

countries_s = [
    '(USA OR America)',
    '(UK OR British OR United Kingdom OR Britain)',
    '(CANADA OR CANADIAN)',
    '(AUSTRALIA)',
    '(Ukraine)',
    '(RUSSIA)',
    '(FRANCE OR FRENCH)',
    '(GERMAN)',
    '(Brazil)',
    '(China OR chinese)',
    '(Japan)',
    '(Pakistan)',
    '(North Korea)',
    '(South Korea)',
    '(India)',
    '(Taiwan)',
    '(NetherLands OR Holland OR Dutch)',
    '(SPAIN OR Spanish)',
    '(Sweden OR Swedish)',
    '(Mexic)',
    '(IRAN)',
    '(ISRAEL)',
    '(Saudi)',
    '(Syria)',
    '(Finland OR FINNISH)',
    '(IRELAND OR IRISH)',
    '(AUSTRIA)',
    '(NORWAY OR Norwegian)',
    '(Switzerland OR swiss)',
    '(ITALY OR ITALIAN)',
    '(MALAYSIA)',
    '(EGYPT)',
    '(TURKEY OR TURKISH)',
    '(portugal OR portuguese)',
    '(Palestin OR West Bank OR GAZA)',
    '(UAE OR United Arab Emirates OR emarat)'
]

countries = [
    'US', 'GB', 'CA', 'AU', 'UA', 'RU', 'FR', 'DE', 'BR', 'CN', 'JP', 'PK',
    'KP', 'KR', 'IN', 'TW', 'NL', 'ES', 'SE', 'MX', 'IR', 'IL', 'SA', 'SY',
    'FI', 'IE', 'AT', 'NO', 'CH', 'IT', 'MY', 'EG', 'TR', 'PT', 'PS', 'AE'
]

data_header = ['Date']
for country in countries:
    data_header.append('WAR/CONFLICT ' + country)

conflicts = [data_header]

with open('ACA.csv', 'a', newline="") as f:
    for year in range(2011, 2023):
        for month in months:

            # Data before July 2011 are not included
            if int(month) < 7 and year == 2011:
                continue

            # Calculate days of the month
            days = 31
            if any([
                month == '04',
                month == '06',
                month == '09',
                month == '11'
            ]):
                days = 30
            if month == '02' and is_leap_year(year):
                days = 29
            if month == '02' and not is_leap_year(year):
                days = 28

            conflict = [month + '/' + str(year)]
                
            c_index = -1
            for country in countries:
                c_index += 1
                country_s = countries_s[c_index]

                # Inputs for the request
                bearer_token = auth()
                headers = create_headers(bearer_token)
                keyword = "(" + country_s + " WAR MILITARY) OR (" + \
                          country_s + " WAR ARMED FORCE) OR (" + country_s + \
                          " CONFLICT POLITIC) OR (" + country_s + \
                          " MILITARY ATTACK) OR (" + country_s + \
                          " ARMED FORCE ATTACK) lang:en"

                start_time = f'{year}-{month}-01T00:00:00.000Z'
                end_time = f"{year}-{month}-{days}T23:59:59.000Z"
                max_results = 400

                count = 0
                flag = True
                next_token = None

                while flag:
                    if count >= 200_000:
                        count = 200_000
                        break

                    # Create query parameters
                    print('creating URL....')
                    url, params = \
                        create_url(keyword, start_time, end_time, max_results)

                    print('getting json response...')
                    json_response = \
                        connect_to_endpoint(url, headers, params, next_token)
                    result_count = json_response['meta']['result_count']
                   
                    if 'next_token' in json_response['meta']:

                        # Save the token to use for next call
                        next_token = json_response['meta']['next_token']
                        print('Next Token: ', next_token)

                        if all([
                            result_count is not None,
                            result_count > 0,
                            next_token is not None
                        ]):
                            count += result_count
                            print(
                                "--------NEXT PAGE-----------",
                                country,
                                month + '/' + str(year) + ':',
                                count
                            )

                            time.sleep(6)

                    # If no next token exists
                    else:
                        if result_count is not None and result_count > 0:
                            print('-------------------')
                            count += result_count
                            print('------FINISHED-------------')

                            print(
                                'Total number of results for: ',
                                country,
                                month + '/' + str(year) + ':',
                                count
                            )

                            time.sleep(5)

                        # This is the final request. Set flag to false & move
                        # to the next time period.
                        flag = False
                        next_token = None
                    time.sleep(5)

                conflict.append(count)
            conflicts.append(conflict)
            
            writer_object = writer(f)
            writer_object.writerow(conflict) 
            f.flush()
f.close()
