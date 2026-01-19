import json
import pandas as pd

from elsapy.elsclient import ElsClient
from elsapy.elssearch import ElsSearch

# Load Elsevier API key
with open("config.json") as config_file:
    config = json.load(config_file)

# Initialise client
client = ElsClient(config["apikey"])

# List for querying Elsevier (PT list)
solution_list = [
    '"BLOCKCHAIN"',
    '"ACCESS CONTROL"',
    'ENCRYPTION',
    '"SUPPLY CHAIN" AND "RISK MANAGEMENT"',
    '"IDENTITY MANAGEMENT"',
    '"Double Patterning Lithography"',
    '"MACHINE LEARNING"',
    '"ANOMALY DETECTION"',
    'CRYPTOGRAPHY',
    '"PENETRATION TESTING"',
    '"intrusion detection" OR "intrusion prevention"',
    '"STATIC ANALYSIS"',
    '"DYNAMIC ANALYSIS"',
    '"MULTI FACTOR AUTHENTICATION" OR "MULTIFACTOR AUTHENTICATION" OR "MULTI-FACTOR AUTHENTICATION"',
    '"LEAST PRIVILEGE"',
    '"SESSION MANAGEMENT"',
    'CAPTCHA',
    'BLACKLISTING',
    '"RATE LIMITING"',
    '"GRAPHICAL MODEL*"',
    'HONEYPOT',
    '"SOFTWARE DEFINED NETWORK"',
    '"GAME THEORY"',
    '"GRAPH MACHINE LEARNING" OR "GRAPH-BASED MACHINE LEARNING"',
    '"IP WHITELIST*"',
    '"TRAFFIC SHAPING"',
    '"PACKET FILTERING"',
    'BLACKHOLING OR "BLACK HOLING" OR ("Black hol*" AND network)',
    '"RANK CORRELATION"',
    'D3NS AND DNS',
    '"SESSION ID RANDOMIZATION" OR "SESSION ID RANDOMISATION"',
    '"STRONG AUTHENTICATION" OR Kerberos',
    '"SECURE SOCKETS LAYER" OR "Transport Layer Security"',
    '{HTTPS}',
    '"CONTINUOUS AUTHENTICATION"',
    '"Identity-Based Encryption" OR "Identity Based Encryption"',
    '"DATA SANITIZATION" OR "DATA SANITISATION"',
    '"OUTLIER DETECTION"',
    '"DATA PROVENANCE"',
    '"ADVERSARIAL TRAINING" OR "adversarial learning"',
    '"TRUSTWORTHY AI"',
    '"DEEP PROBABILISTIC MODEL*"',
    '"Bayesian Network*"',
    '"TROJAN ISOLATION"',
    '"HARDWARE SANDBOXING"',
    '"BEHAVIOR BASED DETECTION" OR "BEHAVIOR-BASED DETECTION" OR "BEHAVIOUR BASED DETECTION" OR "BEHAVIOUR-BASED DETECTION"',
    '"FORMAL VERIFICATION"',
    '"SPLIT MANUFACTURING"',
    '"VULNERABILITY MANAGEMENT"',
    '"FILE INTEGRITY" AND MONITOR*',
    '{VPN}',
    '"CSS MATCHING"',
    '"URI MATCHING"',
    '"PRIVACY PRESERVING"',
    '"SECURE BOOT"',
    '"MERKLE SIGNATURE"',
    '"LIVENESS DETECTION"',
    '"AUDIO DEEPFAKE DETECTION"',
    '"3D FACE RECONSTRUCTION"',
    'BIOMETRICS',
    '"DIGITAL WATERMARK"',
    '"APPLICATION WHITELISTING"',
    '"DATA BACKUP"',
    '"HIDDEN MARKOV MODEL"',
    '"PATCH MANAGEMENT"',
    '"DATA AUGMENTATION"',
    '"DIMENSIONALITY REDUCTION"',
    '"DEFENSIVE DISTILLATION"',
    '"GRADIENT MASKING"',
    '{RRAM}',
    '"SPATIAL SMOOTHING"',
    '"NOISE INJECTION"',
    '"TAINT ANALYSIS"',
    '"NETWORK SEGMENTATION"',
    '"USER BEHAVIOR ANALYTICS" OR "USER BEHAVIOUR ANALYTICS" OR ("USER BEHAVIOR" AND ANALYTICS) OR ("USER BEHAVIOUR" AND ANALYTICS)',
    '"DECEPTION TECHNOLOGY"',
    '"RISK ASSESSMENT"',
    '"LOG CORRELATION"',
    '"DYNAMIC RESOURCE MANAGEMENT"',
    '"SANDBOXING"',
    '"DARKNET MONITORING" OR "MONITOR* DARKNET" OR "DARKNET MONITOR*"',
    '"VIRTUAL KEYBOARD*"',
    '"CODE SIGN*"',
    '"FILE SIGNATURE"',
    '"PUBLIC KEY INFRASTRUCTURE"',
    '"VOICEPRINT AUTHENTICATION"',
    '"MUTUAL AUTHENTICATION"',
    '"PASSWORD SALT*"',
    '"ONE TIME PASSWORD"',
    '"DYNAMIC BINARY INSTRUMENTATION"',
    '"ORTHOGONAL OBFUSCATION"',
    '"PASSWORD HASH*"',
    '{DNSSEC}',
    '"CERTIFICATE PINNING"',
    '"SECURE SIMPLE PAIRING"',
    '"VULNERABILITY ASSESSMENT"',
    '{SIEM}',
    '"STANDARDIZED COMMUNICATION" OR "STANDARDISED COMMUNICATION"',
    '"Control Flow Integrity"',
    '"VULNERABILITY SCAN*"',
    '"PASSWORD STRENGTH METER*"',
    '"PASSWORD MANAGEMENT"',
    '"PASSWORD POLICY" OR "PASSWORD POLICI*"',
    '"GRAPHICAL AUTHENTICATION"',
    '"DATA LOSS PREVENTION"',
    '"DATA LEAKAGE DETECTION" OR "DATA LEAKAGE PREVENTION"',
    '"ACTIVITY MONITORING"',
    '"MOVING TARGET DEFEN*"',
    '"KEYSTROKE DYNAMICS"',
    '"ATTACK TREE"',
    '"Automatic Violation Prevention"',
    '"DISTRIBUTED LEDGER*"',
    '"SOURCE IDENTIFICATION"',
    '"IMAGE RECOGNITION"',
    '{MAUVE}',
    '"HYPERGAME"',
    '"PREBUNKING"',
    '"natural language processing" OR "language model*"'
]

# List for building dataframe column names
column_list = [
    "BLOCKCHAIN",
    "ACCESS_CONTROL",
    "ENCRYPTION",
    "SUPPLY_CHAIN_RISK_MANAGEMENT",
    "IDENTITY_MANAGEMENT",
    "DOUBLE_PATTERNING_LITHOGRAPHY",
    "ML/DL",
    "ANOMALY_DETECTION",
    "CRYPTOGRAPHY",
    "PENETRATION_TESTING",
    "IDS/IPS",
    "STATIC_ANALYSIS",
    "DYNAMIC_ANALYSIS",
    "MULTI_FACTOR_AUTHENTICATION",
    "LEAST_PRIVILEGE",
    "SESSION_MANAGEMENT",
    "CAPTCHA",
    "BLACKLISTING",
    "RATE_LIMITING",
    "GRAPHICAL_MODEL",
    "HONEYPOT",
    "SOFTWARE_DEFINED_NETWORK",
    "GAME_THEORY",
    "GRAPH_BASED_ML",
    "WHITELIST",
    "TRAFFIC_SHAPING",
    "PACKET_FILTERING",
    "BLACKHOLING",
    "RANK_CORRELATION",
    "D3NS",
    "ID_RANDOMIZATION",
    "STRONG_AUTHENTICATION",
    "SSL/TLS",
    "HTTPS",
    "CONTINUOUS_AUTHENTICATION",
    "IDENTITY_BASED_ENCRYPTION",
    "DATA_SANITIZATION",
    "OUTLIER_DETECTION",
    "DATA_PROVENANCE",
    "ADVERSARIAL_TRAINING",
    "TRUSTWORTHY_AI",
    "DEEP_PROBABILISTIC_MODEL",
    "BAYESIAN_NETWORK",
    "TROJAN_ISOLATION",
    "HARDWARE_SANDBOXING",
    "BEHAVIOR_BASED_DETECTION",
    "FORMAL_VERIFICATION",
    "SPLIT_MANUFACTURING",
    "VULNERABILITY_MANAGEMENT",
    "FILE_INTEGRITY_MONITORING",
    "VPN",
    "CSS_MATCHING",
    "URI_MATCHING",
    "PRIVACY_PRESERVING",
    "SECURE_BOOT",
    "MERKLE_SIGNATURE",
    "LIVENESS_DETECTION",
    "AUDIO_DEEPFAKE_DETECTION",
    "3D_FACE_RECONSTRUCTION",
    "BIOMETRICS",
    "DIGITAL_WATERMARK",
    "APPLICATION_WHITELISTING",
    "DATA_BACKUPS",
    "HIDDEN_MARKOV_MODEL",
    "PATCH_MANAGEMENT",
    "DATA_AUGMENTATION",
    "DIMENSIONALITY_REDUCTION",
    "DEFENSIVE_DISTILLATION",
    "GRADIENT_MASK",
    "RRAM",
    "SPATIAL_SMOOTHING",
    "NOISE_INJECTION",
    "TAINT_ANALYSIS",
    "NETWORK_SEGMENTATION",
    "USER_BEHAVIOR_ANALYTICS",
    "DECEPTION_TECHNOLOGY",
    "RISK_ASSESSMENT",
    "LOG_CORRELATION",
    "DYNAMIC_RESOURCE_MANAGEMENT",
    "SANDBOXING",
    "DARKNET_MONITORING",
    "VIRTUAL_KEYBOARDS",
    "CODE_SIGNING",
    "FILE_SIGNATURE",
    "PUBLIC_KEY_INFRASTRUCTURE",
    "VOICEPRINT_AUTHENTICATION",
    "MUTUAL_AUTHENTICATION",
    "PASSWORD_SALT",
    "ONE_TIME_PASSWORD",
    "DYNAMIC_BINARY_INSTRUMENTATION",
    "ORTHOGONAL_OBFUSCATION",
    "PASSWORD_HASHING",
    "DNSSEC",
    "CERTIFICATE_PINNING",
    "SECURE_SIMPLE_PAIRING",
    "VULNERABILITY_ASSESSMENT",
    "SIEM",
    "STANDARDIZED_COMMUNICATION",
    "CONTROL_FLOW_INTEGRITY",
    "VULNERABILITY_SCANNER",
    "PASSWORD_STRENGTH_METERS",
    "PASSWORD_MANAGEMENT",
    "PASSWORD_POLICY",
    "GRAPHICAL_AUTHENTICATION",
    "DATA_LOSS_PREVENTION",
    "DATA_LEAKAGE_DETECTION/PREVENTION",
    "ACTIVITY_MONITORING",
    "MOVING_TARGET_DEFENSE",
    "KEYSTROKE_DYNAMICS",
    "ATTACK_TREE",
    "AUTOMATIC_VIOLATION_PREVENTION",
    "DISTRIBUTED_LEDGERS",
    "SOURCE_IDENTIFICATION",
    "IMAGE_RECOGNITION",
    "MAUVE",
    "HYPERGAME",
    "PREBUNKING",
    "NLP/LLM"
]

# Date range
START_DATE = "2011-07"
END_DATE = "2025-05"
date_range = pd.period_range(START_DATE, END_DATE, freq="M")

years_list = list(range(2011, 2012))
month_list = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]

search_results_list = []
df_columns = [f"Solution_{solution}_Papers" for solution in column_list]

# Collect URLs of all relevant documents (for each month and each PT)
# The search is within the title, abstract, and keywords
for year in years_list:
    print("searching in year:", str(year))
    for month in month_list[6:8]:
        # Date range: July 2011 - May 2025
        if any([
            year == 2011 and month in month_list[:6],
            year == 2025 and month in month_list[6:]
        ]):
            continue

        # List of lists (each list holds monthly results for a type of solution)
        slsm = []
        print("searching in year/month:", str(year), "/", month)

        for index, solution in enumerate(solution_list):
            # List of dictionaries (each dictionary represents a document)
            slm = []
            print("searching for ", solution)

            query = (
                f"(TITLE({solution}) OR ABS({solution}) OR KEY({solution})) "
                "AND (TITLE(*security OR secure OR secur*) "
                "OR ABS(*security OR secure OR secur*) "
                "OR KEY(*security OR secure OR secur*)) "
                f"AND (PUBDATETXT({month} {str(year)}))"
            )

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
