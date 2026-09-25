from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import patheffects
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.textpath import TextPath
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
GRAPH_CSV = ROOT / "B-MTGNN" / "data" / "graph.csv"
LEGACY_GAP_DIR = ROOT / "B-MTGNN" / "model" / "Bayesian" / "forecast" / "gap"
STATE_JSONL = ROOT / "Multi-Agent" / "Results" / "Stage0" / "Forecast" / "node_state_series.jsonl"
OUTPUT = ROOT / "figure" / "threat_pmt_gap_network"

THREAT_FILL = "#4E86A8"
THREAT_EDGE = "#2F6485"
PMT_FILL = "#AFDDB7"
PMT_EDGE = "#659F72"

WIDENING_COLOR = "#9A5F57"
NARROWING_COLOR = "#477F98"

MAX_NODE_SIZE = 371.8 * 5.0
MIN_NODE_SIZE = 26.0 * 5.0
NODE_SIZE_EXPONENT = 0.50
MAJOR_THREAT_COUNT = 10
MAJOR_PMT_COUNT = 12
REPRESENTATIVE_THREATS = {"DDoS", "Malware"}
MIN_GAP_DISTANCE = 0.34
GAP_DISTANCE_SPAN = 0.95
SAME_KIND_COLLISION_WEIGHT = 64.0
MIXED_KIND_COLLISION_WEIGHT = 16.0
SAME_KIND_VISUAL_SPACING = 3.6
MIXED_KIND_VISUAL_SPACING = 1.8
VISUAL_SPACING_WEIGHT = 2.5
EDGE_DISTANCE_WEIGHT = 96.0
LAYOUT_SEEDS = (4,)
MIN_GAP_DISTANCE_RANK_CORRELATION = 0.99
MAX_RELATIVE_GAP_DISTANCE_RMSE = 0.015
MIN_NODE_CLEARANCE_RATIO = 0.95
HUB_ANGULAR_SPREAD_WEIGHT = 7.0
HUB_DIRECTION_BALANCE_WEIGHT = 3.0
EDGE_NODE_CLEARANCE_SCALE = 1.10
EDGE_NODE_CLEARANCE_WEIGHT = 72.0

THREAT_CODES = {
    "Phishing": "PH", "Password Attack": "PA", "Trojan": "TR",
    "Vulnerability": "VU", "Advanced persistent threat": "APT",
    "Disinformation/Misinformation": "DM", "Targeted Attack": "TA",
    "Backdoor": "BD", "Cryptojacking": "CJ", "DNS Spoofing": "DNS",
    "Insider Threat": "IT", "Data Poisoning": "DP", "Session Hijacking": "SH",
    "DDoS": "DD", "Ransomware": "RW", "Account Hijacking": "AH",
    "Zero-day": "ZD", "Malware": "MW", "Brute Force Attack": "BF",
    "Botnet": "BN", "MITM": "MI", "Dropper": "DR",
    "Adversarial Attack": "AA", "Deepfake": "DF", "Supply Chain": "SC",
    "IoT Device Attack": "IoT",
}

PMT_CODES = {
    "ML/DL": "ML", "Penetration Testing": "PT", "NLP/LLM": "NLP",
    "IDS/IPS": "IDS", "Anomaly Detection": "AD", "Cryptography": "CR",
    "Access Control": "AC", "Activity Monitoring": "AM",
    "Adversarial Training": "AT", "Application Whitelisting": "AW",
    "Blacklisting": "BL", "Blockchain": "BC", "CAPTCHA": "CA",
    "Data Provenance": "DP", "Distributed Ledgers": "DL",
    "Encryption": "EN", "Hidden Markov Model": "HMM", "Honeypot": "HP",
    "Taint Analysis": "TA", "Vpn": "VPN",
}


def clean_threat(name: str) -> str:
    return name.removeprefix("Mentions-").removesuffix("-ALL")


def clean_pmt(name: str) -> str:
    return (
        name.removeprefix("Solution_")
        .removesuffix("_Mentions")
        .replace("_", " ")
        .title()
        .replace("Ml/Dl", "ML/DL")
        .replace("Nlp/Llm", "NLP/LLM")
        .replace("Ids/Ips", "IDS/IPS")
        .replace("Captcha", "CAPTCHA")
    )


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()


def legacy_slug(value: str) -> str:
    return (
        slug(value)
        .replace("BEHAVIOUR", "BEHAVIOR")
        .replace("SANITISATION", "SANITIZATION")
        .replace("STANDARDISED", "STANDARDIZED")
    )


def load_graph():
    rows = list(csv.reader(GRAPH_CSV.open(encoding="utf-8")))
    threats = [row[0] for row in rows if row]
    edges = [(row[0], pmt) for row in rows if row for pmt in row[1:] if pmt]
    pmts = sorted({pmt for _, pmt in edges})
    graph = nx.Graph()
    graph.add_nodes_from(threats, kind="threat")
    graph.add_nodes_from(pmts, kind="pmt")
    graph.add_edges_from(edges)
    degree = Counter(node for edge in edges for node in edge)
    return graph, threats, pmts, edges, degree


def load_gap_values(edges):
    by_pair = {}
    for path in LEGACY_GAP_DIR.glob("*_gap.csv"):
        threat = legacy_slug(path.name.removesuffix("_gap.csv"))
        rows = list(csv.reader(path.open(encoding="utf-8")))[1:]
        for row in rows:
            by_pair[(threat, legacy_slug(row[0]))] = np.array(
                [float(value) for value in row[1:4]],
                dtype=float,
            )

    mean_gap = {}
    slope = {}
    for threat, pmt in edges:
        key = (
            legacy_slug(clean_threat(threat)),
            legacy_slug(clean_pmt(pmt)),
        )
        gaps = by_pair.get(key)
        if gaps is None:
            continue
        years = np.array([2025.0, 2026.0, 2027.0])
        mean_gap[(threat, pmt)] = float(gaps.mean())
        slope[(threat, pmt)] = float(np.polyfit(years, gaps, 1)[0])

    return mean_gap, slope


def select_major_subgraph(threats, pmts, edges, degree, gap):
    selected_threats = set(sorted(
        threats,
        key=lambda node: (-degree[node], clean_threat(node)),
    )[:MAJOR_THREAT_COUNT])
    selected_pmts = set(sorted(
        pmts,
        key=lambda node: (-degree[node], clean_pmt(node)),
    )[:MAJOR_PMT_COUNT])

    selected_threats.update(
        node for node in threats
        if clean_threat(node) in REPRESENTATIVE_THREATS
    )

    selected_edges = [
        edge for edge in edges
        if edge in gap
        and edge[0] in selected_threats
        and edge[1] in selected_pmts
    ]
    return selected_threats, selected_pmts, selected_edges


def edge_alpha(edge, slope, selected_edges):
    magnitudes = np.array([abs(slope[item]) for item in selected_edges], dtype=float)
    scale = float(np.quantile(magnitudes, 0.90))
    if scale <= 0.0:
        return 0.50
    normalized = min(abs(slope[edge]) / scale, 1.0)
    return 0.24 + 0.58 * normalized ** 0.85


def load_node_activity():
    activity = defaultdict(list)
    with STATE_JSONL.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["phase"] != "forecast":
                continue
            activity[row["node_id"]].append(max(0.0, float(row["value_raw"])))

    return {
        node_id: sum(values) / len(values)
        for node_id, values in activity.items()
    }


def node_activity_id(node: str, kind: str) -> str:
    name = clean_threat(node) if kind == "threat" else clean_pmt(node)
    prefix = "THREAT" if kind == "threat" else "PMT"
    return f"{prefix}_{slug(name)}"


def activity_sizes(nodes, kind: str, activity):
    values = [activity[node_activity_id(node, kind)] for node in nodes]
    maximum = max(values)
    return [
        MIN_NODE_SIZE
        + (MAX_NODE_SIZE - MIN_NODE_SIZE) * (value / maximum) ** NODE_SIZE_EXPONENT
        for value in values
    ]


def pmt_code(node: str) -> str:
    clean = clean_pmt(node)
    if clean in PMT_CODES:
        return PMT_CODES[clean]
    words = re.findall(r"[A-Za-z0-9]+", clean)
    return "".join(word[0] for word in words).upper()[:5]


LABEL_FONT = FontProperties(weight="bold")
CALLOUT_TANGENT = {"APT": 7.0, "DT": -11.0, "MFA": 11.0}


def inside_label_fontsize(node_size: float, text: str, preferred: float, minimum: float):
    diameter = 2.0 * math.sqrt(node_size / math.pi)
    fontsize = preferred
    while fontsize >= minimum:
        bounds = TextPath((0, 0), text, size=fontsize, prop=LABEL_FONT).get_extents()
        if bounds.width <= diameter * 0.78 and bounds.height <= diameter * 0.68:
            return fontsize
        fontsize -= 0.2
    return None


def annotate_outward(ax, pos, node, text, center, color, fontsize):
    x, y = pos[node]
    dx = x - center[0]
    dy = y - center[1]
    length = math.hypot(dx, dy) or 1.0
    tangent = CALLOUT_TANGENT.get(
        text,
        (sum(ord(char) for char in node) % 7 - 3) * 2.2,
    )
    ox = 14.5 * dx / length - tangent * dy / length
    oy = 14.5 * dy / length + tangent * dx / length
    annotation = ax.annotate(
        text,
        xy=(x, y),
        xytext=(ox, oy),
        textcoords="offset points",
        ha="left" if ox >= 0 else "right",
        va="bottom" if oy >= 0 else "top",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        zorder=6,
        arrowprops={
            "arrowstyle": "-",
            "color": color,
            "linewidth": 0.40,
            "alpha": 0.64,
            "shrinkA": 1.5,
            "shrinkB": 2.5,
        },
    )
    annotation.set_path_effects([
        patheffects.withStroke(linewidth=1.8, foreground="white")
    ])
    return annotation


def orient_landscape(pos, graph):
    nodes = list(pos)
    points = np.array([pos[node] for node in nodes], dtype=float)
    points -= points.mean(axis=0)
    hubs = sorted(nodes, key=lambda node: (-graph.degree[node], str(node)))[:2]
    first_hub, second_hub = hubs
    first_point = points[nodes.index(first_hub)]
    second_point = points[nodes.index(second_hub)]
    hub_axis = second_point - first_point
    angle = -math.atan2(hub_axis[1], hub_axis[0])
    c = math.cos(angle)
    s = math.sin(angle)
    rotated = {
        node: (point[0] * c - point[1] * s, point[0] * s + point[1] * c)
        for node, point in zip(nodes, points)
    }
    if rotated[first_hub][0] > rotated[second_hub][0]:
        rotated = {node: (-x, y) for node, (x, y) in rotated.items()}
    return rotated


def segments_cross(a, b, c, d):
    def cross(first, second, third):
        return (
            (second[0] - first[0]) * (third[1] - first[1])
            - (second[1] - first[1]) * (third[0] - first[0])
        )

    ab_c = cross(a, b, c)
    ab_d = cross(a, b, d)
    cd_a = cross(c, d, a)
    cd_b = cross(c, d, b)
    return ab_c * ab_d < 0.0 and cd_a * cd_b < 0.0


def point_segment_distance(point, start, end):
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-15:
        return np.linalg.norm(point - start)
    t = float(np.dot(point - start, segment) / denominator)
    t = min(1.0, max(0.0, t))
    closest = start + t * segment
    return np.linalg.norm(point - closest)


def layout_metrics(pos, graph, edges, gap, node_sizes, targets):
    edge_distances = np.array([
        math.dist(pos[first], pos[second])
        for first, second in edges
    ])
    target_distances = np.array([targets[edge] for edge in edges])
    gap_values = np.array([gap[edge] for edge in edges])

    correlation = float(spearmanr(gap_values, edge_distances).statistic)
    relative_rmse = float(
        np.sqrt(np.mean((edge_distances - target_distances) ** 2))
        / np.mean(target_distances)
    )

    nodes = list(graph.nodes())
    overlap_count = 0
    overlap_amount = 0.0
    min_node_clearance_ratio = math.inf
    min_nonedge_clearance = math.inf
    for first_index, first in enumerate(nodes):
        for second in nodes[first_index + 1:]:
            distance = math.dist(pos[first], pos[second])
            required = 0.0028 * (
                math.sqrt(node_sizes[first]) + math.sqrt(node_sizes[second])
            )
            penetration = required - distance
            min_node_clearance_ratio = min(
                min_node_clearance_ratio,
                distance / required,
            )
            if penetration > 0.0:
                overlap_count += 1
                overlap_amount += penetration
            if not graph.has_edge(first, second):
                min_nonedge_clearance = min(
                    min_nonedge_clearance,
                    distance / required,
                )

    crossings = 0
    for first_index, first_edge in enumerate(edges):
        for second_edge in edges[first_index + 1:]:
            if set(first_edge) & set(second_edge):
                continue
            if segments_cross(
                pos[first_edge[0]],
                pos[first_edge[1]],
                pos[second_edge[0]],
                pos[second_edge[1]],
            ):
                crossings += 1

    edge_node_crossings = 0
    edge_node_penetration = 0.0
    min_edge_node_clearance_ratio = math.inf
    for first, second in edges:
        start = np.asarray(pos[first], dtype=float)
        end = np.asarray(pos[second], dtype=float)
        for node in nodes:
            if node in (first, second):
                continue
            radius = 0.0028 * math.sqrt(node_sizes[node])
            distance = float(point_segment_distance(
                np.asarray(pos[node], dtype=float),
                start,
                end,
            ))
            min_edge_node_clearance_ratio = min(
                min_edge_node_clearance_ratio,
                distance / radius,
            )
            if distance < radius:
                edge_node_crossings += 1
                edge_node_penetration += radius - distance

    hub_min_angle = math.pi
    for hub in nodes:
        neighbors = list(graph.neighbors(hub))
        if len(neighbors) < 3:
            continue
        angles = sorted(
            math.atan2(
                pos[neighbor][1] - pos[hub][1],
                pos[neighbor][0] - pos[hub][0],
            )
            % (2.0 * math.pi)
            for neighbor in neighbors
        )
        gaps = [
            angles[index + 1] - angles[index]
            for index in range(len(angles) - 1)
        ]
        gaps.append(angles[0] + 2.0 * math.pi - angles[-1])
        hub_min_angle = min(hub_min_angle, min(gaps))

    primary_hub = max(nodes, key=lambda node: (graph.degree[node], str(node)))
    primary_neighbors = list(graph.neighbors(primary_hub))
    if len(primary_neighbors) >= 3:
        primary_points = np.array([pos[node] for node in primary_neighbors])
        primary_hub_area = float(ConvexHull(primary_points).volume)
        primary_hub_nn_mean = float(np.mean([
            min(
                math.dist(pos[node], pos[other])
                for other in primary_neighbors
                if other != node
            )
            for node in primary_neighbors
        ]))
    else:
        primary_hub_area = 0.0
        primary_hub_nn_mean = 0.0

    return {
        "correlation": correlation,
        "relative_rmse": relative_rmse,
        "overlap_count": overlap_count,
        "overlap_amount": overlap_amount,
        "min_node_clearance_ratio": min_node_clearance_ratio,
        "crossings": crossings,
        "edge_node_crossings": edge_node_crossings,
        "edge_node_penetration": edge_node_penetration,
        "min_edge_node_clearance_ratio": min_edge_node_clearance_ratio,
        "min_nonedge_clearance": min_nonedge_clearance,
        "hub_min_angle": hub_min_angle,
        "primary_hub_area": primary_hub_area,
        "primary_hub_nn_mean": primary_hub_nn_mean,
    }


def optimize_gap_layout(graph, edges, gap, node_sizes, targets, seed):
    nodes = sorted(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}

    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(nodes)
    layout_graph.add_edges_from(sorted(edges))

    initial = nx.spring_layout(
        layout_graph,
        seed=seed,
        weight=None,
        k=0.42,
        iterations=1200,
        threshold=1e-7,
        scale=1.0,
    )
    initial = np.array([initial[node] for node in nodes], dtype=float)

    def residuals(flat):
        points = flat.reshape((-1, 2))
        values = []
        for edge in edges:
            first = points[index[edge[0]]]
            second = points[index[edge[1]]]
            distance = np.linalg.norm(first - second)
            values.append((distance - targets[edge]) * EDGE_DISTANCE_WEIGHT)

        for first_index in range(len(nodes)):
            for second_index in range(first_index + 1, len(nodes)):
                first = nodes[first_index]
                second = nodes[second_index]
                distance = np.linalg.norm(points[first_index] - points[second_index])
                required = 0.0028 * (
                    math.sqrt(node_sizes[first]) + math.sqrt(node_sizes[second])
                )
                same_kind = graph.nodes[first]["kind"] == graph.nodes[second]["kind"]
                collision_weight = (
                    SAME_KIND_COLLISION_WEIGHT
                    if same_kind
                    else MIXED_KIND_COLLISION_WEIGHT
                )
                values.append(max(0.0, required - distance) * collision_weight)

                if not graph.has_edge(first, second):
                    spacing = (
                        SAME_KIND_VISUAL_SPACING
                        if same_kind
                        else MIXED_KIND_VISUAL_SPACING
                    )
                    visual_required = required * spacing
                    values.append(
                        max(0.0, visual_required - distance) * VISUAL_SPACING_WEIGHT
                    )

        if EDGE_NODE_CLEARANCE_WEIGHT > 0.0:
            for first, second in edges:
                start = points[index[first]]
                end = points[index[second]]
                for node in nodes:
                    if node in (first, second):
                        continue
                    required = (
                        0.0028
                        * math.sqrt(node_sizes[node])
                        * EDGE_NODE_CLEARANCE_SCALE
                    )
                    distance = point_segment_distance(
                        points[index[node]],
                        start,
                        end,
                    )
                    values.append(
                        max(0.0, required - distance)
                        * EDGE_NODE_CLEARANCE_WEIGHT
                    )

        if HUB_ANGULAR_SPREAD_WEIGHT > 0.0:
            for hub in nodes:
                neighbors = list(graph.neighbors(hub))
                if len(neighbors) < 4:
                    continue
                ideal_angle = 2.0 * math.pi / len(neighbors)
                for first_index, first in enumerate(neighbors):
                    first_edge = (hub, first) if (hub, first) in targets else (first, hub)
                    first_radius = targets[first_edge]
                    for second in neighbors[first_index + 1:]:
                        second_edge = (
                            (hub, second)
                            if (hub, second) in targets
                            else (second, hub)
                        )
                        second_radius = targets[second_edge]
                        desired = math.sqrt(max(
                            0.0,
                            first_radius ** 2
                            + second_radius ** 2
                            - 2.0 * first_radius * second_radius * math.cos(ideal_angle),
                        ))
                        distance = np.linalg.norm(
                            points[index[first]] - points[index[second]]
                        )
                        values.append(
                            max(0.0, desired - distance)
                            * HUB_ANGULAR_SPREAD_WEIGHT
                        )

        if HUB_DIRECTION_BALANCE_WEIGHT > 0.0:
            for hub in nodes:
                neighbors = list(graph.neighbors(hub))
                if len(neighbors) < 4:
                    continue
                directions = []
                hub_point = points[index[hub]]
                for neighbor in neighbors:
                    vector = points[index[neighbor]] - hub_point
                    length = np.linalg.norm(vector)
                    if length > 1e-12:
                        directions.append(vector / length)
                if directions:
                    imbalance = np.mean(directions, axis=0)
                    values.extend((imbalance * HUB_DIRECTION_BALANCE_WEIGHT).tolist())


        values.extend((points.mean(axis=0) * 0.1).tolist())
        return np.asarray(values, dtype=float)

    result = least_squares(
        residuals,
        initial.ravel(),
        max_nfev=4000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    points = result.x.reshape((-1, 2))
    return orient_landscape({
        node: (float(points[i, 0]), float(points[i, 1]))
        for i, node in enumerate(nodes)
    }, graph)


def build_gap_layout(graph, edges, gap, node_sizes):
    maximum_gap = max(gap[edge] for edge in edges)
    targets = {
        edge: MIN_GAP_DISTANCE + GAP_DISTANCE_SPAN * gap[edge] / maximum_gap
        for edge in edges
    }

    if len(LAYOUT_SEEDS) == 1:
        return optimize_gap_layout(
            graph,
            edges,
            gap,
            node_sizes,
            targets,
            LAYOUT_SEEDS[0],
        )

    candidates = []
    for seed in LAYOUT_SEEDS:
        pos = optimize_gap_layout(graph, edges, gap, node_sizes, targets, seed)
        metrics = layout_metrics(pos, graph, edges, gap, node_sizes, targets)
        if metrics["correlation"] < MIN_GAP_DISTANCE_RANK_CORRELATION:
            continue
        if metrics["relative_rmse"] > MAX_RELATIVE_GAP_DISTANCE_RMSE:
            continue
        if metrics["min_node_clearance_ratio"] < MIN_NODE_CLEARANCE_RATIO:
            continue
        candidates.append((pos, metrics, seed))

    if not candidates:
        raise RuntimeError("No readable layout preserved the required GAP-distance ranking.")

    candidates.sort(key=lambda item: (
        item[1]["overlap_count"],
        item[1]["edge_node_crossings"],
        item[1]["edge_node_penetration"],
        item[1]["crossings"],
        -item[1]["primary_hub_area"],
        -item[1]["primary_hub_nn_mean"],
        -item[1]["hub_min_angle"],
        item[1]["overlap_amount"],
        -item[1]["min_nonedge_clearance"],
        -item[1]["correlation"],
        item[1]["relative_rmse"],
        item[2],
    ))
    return candidates[0][0]


def render():
    full_graph, full_threats, full_pmts, all_edges, full_degree = load_graph()
    gap, slope = load_gap_values(all_edges)
    selected_threats, selected_pmts, edges = select_major_subgraph(
        full_threats,
        full_pmts,
        all_edges,
        full_degree,
        gap,
    )
    graph = full_graph.edge_subgraph(edges).copy()
    threats = [node for node in full_threats if node in selected_threats and node in graph]
    pmts = [node for node in full_pmts if node in selected_pmts and node in graph]
    degree = Counter(node for edge in edges for node in edge)

    activity = load_node_activity()
    threat_sizes = activity_sizes(threats, "threat", activity)
    pmt_sizes = activity_sizes(pmts, "pmt", activity)
    node_sizes = {
        **dict(zip(threats, threat_sizes)),
        **dict(zip(pmts, pmt_sizes)),
    }
    pos = build_gap_layout(graph, edges, gap, node_sizes)

    fig, ax = plt.subplots(figsize=(7.20, 5.25), facecolor="white")
    ax.set_facecolor("white")

    for edge in edges:
        color = WIDENING_COLOR if slope[edge] >= 0.0 else NARROWING_COLOR
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[edge],
            ax=ax,
            edge_color=color,
            width=1.15,
            alpha=edge_alpha(edge, slope, edges),
        )

    threat_size = dict(zip(threats, threat_sizes))
    pmt_size = dict(zip(pmts, pmt_sizes))

    ax.scatter(
        [pos[node][0] for node in threats],
        [pos[node][1] for node in threats],
        s=threat_sizes,
        facecolor=THREAT_FILL,
        edgecolor=THREAT_EDGE,
        linewidth=0.82,
        alpha=0.96,
        zorder=4,
    )

    ax.scatter(
        [pos[node][0] for node in pmts],
        [pos[node][1] for node in pmts],
        s=pmt_sizes,
        facecolor=PMT_FILL,
        edgecolor=PMT_EDGE,
        linewidth=0.62,
        alpha=0.96,
        zorder=3,
    )

    center = (
        sum(x for x, _ in pos.values()) / len(pos),
        sum(y for _, y in pos.values()) / len(pos),
    )

    for node in threats:
        x, y = pos[node]
        code = THREAT_CODES[clean_threat(node)]
        fontsize = inside_label_fontsize(threat_size[node], code, 9.0, 6.0)
        if fontsize is not None:
            ax.text(
                x,
                y,
                code,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color="white",
                zorder=5,
            )
        else:
            annotate_outward(ax, pos, node, code, center, THREAT_EDGE, 7.0)

    for node in pmts:
        x, y = pos[node]
        code = pmt_code(node)
        fontsize = inside_label_fontsize(pmt_size[node], code, 9.0, 6.0)
        if fontsize is not None:
            ax.text(
                x,
                y,
                code,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color="#24482D",
                zorder=5,
            )
        else:
            annotate_outward(ax, pos, node, code, center, "#315D3B", 7.0)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=THREAT_FILL,
               markeredgecolor=THREAT_EDGE, markersize=6.6,
               label="Threat (size: representative activity, NoI/NoP)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PMT_FILL,
               markeredgecolor=PMT_EDGE, markersize=5.8,
               label="PMT (size: NoP)"),
    ]
    handles.extend([
        Line2D([0, 1], [0, 0], color=WIDENING_COLOR, lw=1.35,
               alpha=0.75, label="Gap increasing"),
        Line2D([0, 1], [0, 0], color=NARROWING_COLOR, lw=1.35,
               alpha=0.75, label="Gap decreasing"),
    ])

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=8.0,
        title="Major Threat–PMT hubs • distance encodes mean gap • opacity reflects |slope|",
        title_fontsize=10.0,
        columnspacing=0.85,
        handlelength=1.45,
        handletextpad=0.35,
    )

    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    ax.margins(x=0.050, y=0.070)
    fig.subplots_adjust(left=0.008, right=0.992, top=0.925, bottom=0.018)

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            OUTPUT.with_suffix(f".{suffix}"),
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)


if __name__ == "__main__":
    render()
