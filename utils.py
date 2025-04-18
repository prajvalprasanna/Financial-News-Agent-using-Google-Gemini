import pandas as pd
import matplotlib.pyplot as plt

def summarize_to_dataframe(summaries):
    return pd.DataFrame([{
        "Title": s["title"],
        "Summary": s["summary"],
        "Link": s["link"]
    } for s in summaries])

def plot_confidence_trend(memory):
    scores = []
    labels = []
    for item in memory:
        if item["role"] == "agent":
            content = item["content"]
            try:
                score = int([s for s in content.split() if s.isdigit()][-1])
                scores.append(score)
                labels.append(len(scores))
            except:
                continue

    if not scores:
        return None

    plt.figure(figsize=(6, 3))
    plt.plot(labels, scores, marker='o')
    plt.title("Confidence Scores Over Time")
    plt.xlabel("Query Number")
    plt.ylabel("Confidence Score")
    plt.grid(True)
    return plt
