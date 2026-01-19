import json
import pandas as pd

from elsapy.elsclient import ElsClient
from elsapy.elssearch import ElsSearch

# Load Elsevier API key
with open("config.json") as config_file:
    config = json.load(config_file)

# Initialise client
client = ElsClient(config["apikey"])

# List to query Elsevier (attack list)
attack_list = [
    '"Denial-of-Service Attack" OR {DDoS}',
    'Phishing',
    'Ransomware',
    'Password Attack',
    '"SQL Injection"',
    '("Account Hijack*") OR ("Account Takeover")',
    'Website Deface*',
    'Trojan Attack',
    'Vulnerabilit* Attack exploit*',
    '"Zero-day" AND Attack',
    '"Advanced Persistent Threat"',
    '"cross-site scripting"',
    'Malware OR TROJAN OR SPYWARE OR MALVERTIS* OR RANSOMWARE OR ("COMPUTER VIRUS") OR (WORM ATTACK) OR (KEYLOG* Attack) OR (KEYSTROKE LOG Attack) OR (MALICIOUS CODE) OR (MALICIOUS SOFTWARE) OR ADWARE OR ROOTKIT OR BOTNET OR (BACKDOOR ATTACK)',
    '"Data Breach" OR "Data Leak*" OR "information leak*" OR "Data SPILL" OR "unintentional information disclos*"',
    '(Disinformation spread) OR (Misinformation spread) OR ("false information" spread) OR ("misleading information" spread) OR Deepfake',
    '"Targeted Attack"',
    'Adware',
    '"Brute Force" Attack',
    'Malvertis*',
    'Backdoor Attack',
    'Botnet',
    'Cryptojack*',
    'Worm Attack',
    'Spyware',
    '"Man-in-the-middle" OR {MITM}',
    '"DNS Spoof*" OR "DNS cache poison*"',
    'Pegasus Spyware',
    'CoolWebSearch',
    'Gator GAIN',
    '"180search Assistant"',
    'Transponder vx2*',
    'WannaCry AND (Ransomware OR Attack)',
    '"Colonial Pipeline" AND (Ransomware OR Attack)',
    'Cryptolocker OR Crypto-locker',
    'Dropper Trojan',
    'Wiper malware',
    'Pharming Attack',
    '"Insider Threat"',
    '(Drive-by Download) OR (Drive-by Install)',
    'Rootkit',
    'Adversarial Attack',
    '"Data Poisoning"',
    'Deepfake',
    'Deeplocker OR (Deep-locker) OR "Deep locker"',
    '"Supply Chain" AND Attack',
    'IoT Device Attack',
    '(KEYLOG* Attack) OR (KEYSTROKE LOG Attack)',
    '"DNS Tunnel*"',
    '"Session Hijacking" OR "cookie hijacking" OR "cookie poisoning"',
    'URL Attack',
    '"Unknown Attack" AND (Security OR Cyber*)'
]

# List for building dataframe column names
column_list = [
    "DDoS",
    "Phishing",
    "Ransomware",
    "Password_Attack",
    "SQL_Injection",
    "Account_Hijacking",
    "Defacement",
    "Trojan",
    "Vulnerability",
    "Zero_day",
    "Advanced_Persistent_Threat",
    "XSS",
    "Malware",
    "Data_Breach",
    "Disinformation/Misinformation",
    "Targeted_Attack",
    "Adware",
    "Brute_Force_Attack",
    "Malvertising",
    "Backdoor",
    "Botnet",
    "Cryptojacking",
    "Worms",
    "Spyware",
    "MITM",
    "DNS_Spoofing",
    "Pegasus_Spyware",
    "CoolWebSearch_Spyware",
    "Gator_GAIN_Spyware",
    "180search_Assistant_Spyware",
    "Transponder_vx2_Spyware"
    "WannaCry_Ransomware",
    "Colonial_Pipeline_Ransomware",
    "Cryptolocker",
    "Dropper",
    "Wiper",
    "Pharming",
    "Insider_Threat",
    "Drive_by",
    "Rootkit",
    "Adversarial_Attack",
    "Data_Poisoning",
    "Deepfake",
    "Deeplocker",
    "Supply_Chain",
    "IoT_Device_Attack",
    "Keylogger",
    "DNS_Tunneling",
    "Session_Hijacking",
    "URL_manipulation",
    "Unknown_Attack"
]

# Date range
START_DATE = "2011-07"
END_DATE = "2025-05"
date_range = pd.period_range(START_DATE, END_DATE, freq="M")

years_list = list(range(2011, 2026))
month_list = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]

empty = "Result set was empty"
search_results_list = []
df_columns = [f"Papers_{attack}" for attack in column_list]

# Collect URLs of all relevant documents (per month and attack type)
# The search is within the title, abstract, and keywords
for year in years_list:
    print("searching in year:", str(year))
    for month in month_list:
        # Date range: July 2011 - May 2025
        if any([
            year == 2011 and month in month_list[:6],
            year == 2025 and month in month_list[6:]
        ]):
            continue

        # List of lists (each list holds monthly results for a type of attack)
        slsm = []
        print("searching in year/month:", str(year), "/", month)

        for index, attack in enumerate(attack_list):
            # List of dictionaries (each dictionary represents a document)
            slm = []
            print("searching for ", attack)

            query = "(TITLE(" + attack + ") OR ABS(" + \
                    attack + ") OR KEY(" + attack + ")) AND PUBDATETXT(" + \
                    month + " " + str(year) + ")"

            # Search for documents
            doc_srch = ElsSearch(query, "scopus")
            doc_srch.execute(client, get_all=True)

            # Get the number of papers found
            n_papers = len(doc_srch.results)
            print(f"Found {n_papers} papers")

            slsm.append(n_papers)
        search_results_list.append(slsm)

# Create dataframe
df = pd.DataFrame(search_results_list, columns=df_columns, index=date_range)
df.index.name = "Date"

# Export as CSV
output_csv = "PTs_NoP.csv"
df.to_csv(output_csv, index=True)
print(f"Saved to {output_csv}")
