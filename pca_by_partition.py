import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import sys
import csv
from helper import safe_read_csv

filename = "fy2025_output.txt"

columns = [
    "SubmitTime","StartTime","EndTime","RunTime","WaitTime","netid","groupName","JobID","JobName","NodeList","NNodes","ReqCPUS","CPUTimeRAW","DerivedExitCode","Timelimit", "State", "Priority", "Partition", "NCPUS", "longGroupName", "schoolName"
]
masked_fields = ['WaitTime','SubmitTime', 'StartTime', 'EndTime','CPUTimeRAW', 'DerivedExitCode', 'State','longGroupName']
expected_col_count = len(columns)

if __name__ == "__main__":
    df = safe_read_csv(filename, columns, expected_col_count)
    print(f"DataFrame loaded with shape: {df.shape}", file=sys.stderr)

    partition_counts = df['Partition'].value_counts(dropna=True)
    print("\nRows per partition:", file=sys.stderr)
    print(partition_counts, file=sys.stderr)

    partitions = partition_counts.index
    writer = csv.writer(sys.stdout)
    writer.writerow(["Partition", "Count", "ExplainedVarPC1", "ExplainedVarPC2", "ExplainedVarPC3", "R-Squared"])

    for part in partitions:
        df_part = df[df['Partition'] == part]
        df_num = df_part.select_dtypes(include='number').dropna()

        if 'WaitTime' not in df_num.columns or df_num.shape[0] < 5:
            continue

        X = df_num.drop(columns=[col for col in masked_fields if col in df_num.columns])
        y = df_num['WaitTime']
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        explained = pca.explained_variance_ratio_.cumsum()

        if X_pca.shape[1] >= 4:
            model = LinearRegression().fit(X_pca[:, :4], y)
            r2 = model.score(X_pca[:, :4], y)
        else:
            r2 = ""

        writer.writerow([part, df_num.shape[0]] + list(explained[:3]) + [r2])

