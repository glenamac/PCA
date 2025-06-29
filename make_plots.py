import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helper import safe_read_csv

def plot_waittime_hist(df, filename="waittime_hist.png", log_scale = True):
    ax = df['WaitTime'].dropna().hist(bins=100)
    if log_scale:
        ax.set_yscale('log')
    plt.xlabel('WaitTime')
    plt.ylabel('Frequency')
    plt.title('Distribution of WaitTime')
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

def plot_waittime_by_partition(df, filename="waittime_by_partition.png", log_scale = True):
    ax = df.boxplot(column='WaitTime', by='Partition', rot=90, grid=False)
    if log_scale:
        ax.set_yscale('log')
    plt.title('WaitTime by Partition')
    plt.suptitle('')
    plt.ylabel('WaitTime')
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

def plot_job_counts_by_partition(df, filename="job_counts_by_partition.png", log_scale=True):
    ax = df['Partition'].value_counts().plot(kind='bar')
    if log_scale:
        ax.set_yscale('log')
    plt.title('Job Counts by Partition')
    plt.xlabel('Partition')
    plt.ylabel('Number of Jobs')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

def plot_corr_heatmap(df, numeric_cols, filename="correlation_matrix.png"):
    corr = df[numeric_cols].corr()
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.colorbar(label='Correlation Coefficient')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

def plot_pairplot(df, numeric_cols, filename="pairplot_numeric.png", sample_size=1000):
    sampled_df = df[numeric_cols].dropna()
    if len(sampled_df) > sample_size:
        sampled_df = sampled_df.sample(n=sample_size, random_state=42)
    sns.pairplot(sampled_df)
    plt.savefig(filename)
    plt.clf()

def plot_runtime_vs_waittime(df, filename="runtime_vs_waittime.png", log_scale=True, bins=100):
    x = df['RunTime'].dropna()/3600.
    y = df['WaitTime'].dropna()/3600.
    x, y = x.align(y, join='inner')

    # Filter zeros and negatives for log scale
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    # Define log-spaced bins
    xbins = np.logspace(np.log10(x.min()), np.log10(x.max()), bins)
    ybins = np.logspace(np.log10(y.min()), np.log10(y.max()), bins)

    plt.figure(figsize=(8, 6))
    plt.hist2d(x, y, bins=[xbins, ybins], cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(label='Job Count')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('RunTime [hr]')
    plt.ylabel('WaitTime [hr]')
    plt.title('RunTime vs WaitTime')
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()



filename = "fy2025_output.txt"

columns = [
    "SubmitTime", "StartTime", "EndTime", "RunTime", "WaitTime", "netid", "groupName",
    "JobID", "JobName", "NodeList", "NNodes", "ReqCPUS", "CPUTimeRAW", "DerivedExitCode",
    "Timelimit", "State", "Priority", "Partition", "NCPUS", "longGroupName", "schoolName"
]

expected_col_count = len(columns)

if __name__ == "__main__":
    df = safe_read_csv(filename, columns, expected_col_count)

    plot_waittime_hist(df)
    plot_waittime_by_partition(df)
    plot_runtime_vs_waittime(df)
    plot_job_counts_by_partition(df)

    numeric_cols = ['WaitTime', 'RunTime', 'NCPUS', 'ReqCPUS', 'NNodes', 'Priority', 'Timelimit']
    plot_corr_heatmap(df, numeric_cols)
#    plot_pairplot(df, numeric_cols)

