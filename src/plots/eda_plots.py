import pandas as pd

def plot_score_distribution(df, percent=False):
    plot_df = pd.DataFrame({"Total":df["Score"].value_counts()})
    plot_df["Total"] = plot_df["Total"].astype(int)
    plot_df = plot_df.reset_index()
    plot_df.rename(columns={"index":"Rating"},inplace=True)
    plot_df = plot_df.sort_values(by="Score")
    
    if percent:
        plot_df["Percent"] = plot_df["Total"].apply(lambda x: x/plot_df["Total"].sum()*100)
        plot_df.plot.bar(x="Score",y="Percent",title="Score Distribution", rot=0)
    else:
        plot_df.plot.bar(x="Score",y="Total",title="Score Distribution", rot=0)