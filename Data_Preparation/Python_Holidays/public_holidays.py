"""
Created on Jan 20 01:49:41 2023

This script produces a csv file "public_holidays.csv" with the monthly number
of public holidays in 36 countries in the period July 2011 until March 2025.

Code is based on python holidays package:
https://holidays.readthedocs.io/en/latest/

Version required to match original dataset: holidays==0.13

@author: Zaid Almahmoud
"""
from datetime import date
from itertools import product
import holidays
import csv

START_DATE = date(2011, 7, 1)
END_DATE = date(2025, 3, 1)


def is_leap_year(curr_year: int) -> bool:
    """
    Determine whether the year is a leap year.

    :param curr_year: The year in question as integer.
    :return: Boolean indicator of the whether input year is leap.
    """
    return curr_year % 4 == 0 and (curr_year % 100 != 0 or curr_year % 400 == 0)


def within_date_range(curr_month: str, curr_year: int) -> bool:
    """
    Check whether the current month and year are within the required date range.

    :param curr_month: String representation of the month e.g., '01', '10' etc.
    :param curr_year: Current year as integer.

    :return: Boolean indicator for falling within the required date range.
    """
    input_date = date(curr_year, int(curr_month), 1)
    return START_DATE <= input_date <= END_DATE


def date_is_valid(curr_day: int, curr_month: str, curr_year: int) -> bool:
    """
    Check is a date is valid. Examples:
    date_is_valid(31, "04", 2022) -> False (April has only 30 days).
    date_is_valid(29, "02", 2025) -> False (2025 is not a leap year).
    date_is_valid(2, "02", 2021) -> True.

    :param curr_day: Day in question as integer.
    :param curr_month: Month in question as string e.g., "02", "10" etc.
    :param curr_year: Year in question as integer.

    :return: Boolean indicator for valid date.
    """
    if curr_month == "02" and curr_day > 28 and not is_leap_year(curr_year):
        return False
    if curr_month == "02" and curr_day > 29:
        return False
    if curr_day > 30 and any([
        curr_month == "04",
        curr_month == "06",
        curr_month == "09",
        curr_month == "11"
    ]):
        return False
    return True


country_codes = [
    "US", "GB", "CA", "AU", "UA", "RU", "FR", "DE", "BR", "CN", "JP", "PK",
    "KP", "KR", "IN", "TW", "NL", "ES", "SE", "MX", "IR", "IL", "SA", "SY",
    "FI", "IE", "AT", "NO", "CH", "IT", "MY", "EG", "TR", "PT", "PS", "AE"
]

months = [
    "01", "02", "03", "04", "05", "06",
    "07", "08", "09", "10", "11", "12"
]

all_data = []  # List to store holiday data for all countries
missing = set()  # For countries with missing holiday data

if __name__ == "__main__":

    # Init date index list and date range
    date_index = ["date"]
    year_range = range(START_DATE.year, END_DATE.year + 1)
    date_range = tuple(product(year_range, months))

    # Run loop once to build date index (MM/YYYY)
    for year, month in date_range:
        if within_date_range(month, year):
            date_s = month + "/" + str(year)
            date_index.append(date_s)
    all_data.append(date_index)

    # Collect count of public holidays per month for each country
    for country in country_codes:
        country_data = [country + "_holiday"]  # Column name

        # Collect data if the date is within the required date range
        for year, month in date_range:
            if within_date_range(month, year):
                counter = 0

                # Loop through all days of the month
                for day in range(1, 32):
                    if not date_is_valid(day, month, year):
                        continue

                    try:
                        country_holidays = holidays.country_holidays(country)
                    except NotImplementedError:
                        if country not in missing:
                            missing.add(country)
                        continue

                    if date(year, int(month), day) in country_holidays:
                        counter += 1

                country_data.append(counter)
        all_data.append(country_data)
        print("Added holidays of", country)

    with open("PH.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(list(map(list, zip(*all_data))))
