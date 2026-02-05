import json
import numpy as np
import pandas as pd
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go

def make_dataframe_and_clusters(emb_2d, labels, clean_entries):
    df = pd.DataFrame({
        "x": emb_2d[:, 0],
        "y": emb_2d[:, 1],
        "cluster": labels,
        "text": [x["body"] for x in clean_entries],
        "type": [x["type"] for x in clean_entries],
        "author":[x["author"] for x in clean_entries],
    })

    unique_clusters = sorted(set(labels))
    clusters = [c for c in unique_clusters if c != -1] + ([-1] if -1 in unique_clusters else [])
    return df, clusters

def extract_cluster_representatives(df):
    cluster_comments = defaultdict(list)

    for cluster_id in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cluster_id]

        if cluster_df.empty:
            continue

        # --- centroid in 2D space ---
        centroid_x = cluster_df["x"].mean()
        centroid_y = cluster_df["y"].mean()

        # --- distance to centroid ---
        distances = np.sqrt(
            (cluster_df["x"] - centroid_x) ** 2 +
            (cluster_df["y"] - centroid_y) ** 2
        )

        cluster_df = cluster_df.assign(distance=distances)

        # --- take top-k closest ---
        closest = cluster_df.nsmallest(5, "distance")

        key = f"cluster_{cluster_id}" if cluster_id != -1 else "noise_-1"
        cluster_comments[key] = closest["text"].tolist()

    with open("data/cluster_representatives.json", "w", encoding="utf-8") as f:
        json.dump(cluster_comments, f, ensure_ascii=False, indent=2)
    print(f"Cluster representatives saved to data/cluster_representatives.json")


def plot_clusters(df, clusters):
    # === Create figure ===
    fig = go.Figure()
    color_palette = px.colors.qualitative.Plotly
    cluster_to_trace_index = {}

    for i, cluster_id in enumerate(clusters):
        cluster_df = df[df["cluster"] == cluster_id]
        if cluster_df.empty:
            continue
        color = color_palette[i % len(color_palette)]
        trace = go.Scattergl(
            x=cluster_df["x"],
            y=cluster_df["y"],
            mode="markers",
            name=f"Cluster {cluster_id}" if cluster_id != -1 else "Noise (-1)",
            marker=dict(size=7, color=color, opacity=0.8, line=dict(width=0.3, color="black")),
            customdata=np.stack([cluster_df["text"], cluster_df["type"], cluster_df["cluster"], cluster_df["author"]], axis=-1),
            hovertemplate=(
                "<b>Author:</b> %{customdata[3]}<br>"
                "<b>Cluster:</b> %{customdata[2]}<br>"
                "<b>Type:</b> %{customdata[1]}<br>"
                "<b>Text:</b> %{customdata[0]}<extra></extra>"
            ),
            visible=True
        )
        fig.add_trace(trace)
        cluster_to_trace_index[cluster_id] = len(fig.data) - 1

    # === Dropdown menu for clusters ===
    buttons = []

    # Show All
    all_visible = [True] * len(fig.data)
    buttons.append(dict(
        label="Show All",
        method="update",
        args=[{"visible": all_visible}, {"title": "All clusters"}]
    ))

    # Show All (No Noise)
    has_noise = -1 in cluster_to_trace_index
    if has_noise:
        noise_trace_idx = cluster_to_trace_index[-1]
        all_no_noise = all_visible.copy()
        all_no_noise[noise_trace_idx] = False
        buttons.append(dict(
            label="Show All (No Noise)",
            method="update",
            args=[{"visible": all_no_noise}, {"title": "All clusters (no noise)"}]
        ))

    # Individual clusters
    for cluster_id, trace_idx in cluster_to_trace_index.items():
        vis = [False] * len(fig.data)
        vis[trace_idx] = True
        buttons.append(dict(
            label=f"Cluster {cluster_id}" if cluster_id != -1 else "Noise (-1)",
            method="update",
            args=[{"visible": vis}, {"title": f"Cluster {cluster_id}"}]
        ))

    # === Layout ===
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.15,
                xanchor="left",
                y=1.0,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="gray",
                borderwidth=1
            )
        ],
        title="YouTube comments clusterized",
        width=1100,
        height=800,
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font_size=12)
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # === Save & show ===
    fig.write_html("static/plot.html")
    fig.show()