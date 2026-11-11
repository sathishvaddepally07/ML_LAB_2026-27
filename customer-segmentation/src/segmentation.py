import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import silhouette_score

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_customers.csv")

def load_sample_data():
    return pd.read_csv(DATA_PATH)

def _prepare(df):
    if df.empty:
        raise ValueError("The uploaded CSV has no rows.")
    df = df.copy()
    drop_cols = [c for c in df.columns if c.strip().lower() in {"customerid", "customer_id", "id", "name"}]
    df = df.drop(columns=drop_cols, errors="ignore")
    if len(df) < 5:
        raise ValueError("Please provide at least 5 customer records.")
    numeric = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical))
    X = ColumnTransformer(transformers).fit_transform(df)
    if X.shape[1] == 0:
        raise ValueError("No usable features were found.")
    return X, df

def run_analysis(df, k=4):
    X, clean = _prepare(df)
    n = len(clean)
    k = min(k, n-1)
    km_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    gmm_labels = GaussianMixture(n_components=k, random_state=42, reg_covar=1e-5).fit_predict(X)
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(X)
    fa_n = min(2, X.shape[1], n-1)
    fa_xy = FactorAnalysis(n_components=fa_n, random_state=42).fit_transform(X)
    def sil(labels):
        if len(set(labels)) < 2 or len(set(labels)) >= n: return None
        return round(float(silhouette_score(X, labels)), 3)
    display = clean.copy()
    display["KMeans Cluster"] = km_labels + 1
    display["GMM Cluster"] = gmm_labels + 1
    numeric_cols = clean.select_dtypes(include=["number"]).columns.tolist()
    summaries = []
    for c in sorted(np.unique(km_labels)):
        part = display[display["KMeans Cluster"] == c+1]
        row = {"cluster": int(c+1), "customers": int(len(part))}
        for col in numeric_cols[:3]: row[col] = round(float(part[col].mean()), 2)
        summaries.append(row)
    points = [{"x":round(float(x),4),"y":round(float(y),4),"cluster":int(cl+1)} for (x,y),cl in zip(xy,km_labels)]
    return {"count":n,"features":int(X.shape[1]),"kmeans_silhouette":sil(km_labels),
      "gmm_silhouette":sil(gmm_labels),"pca_variance":float(pca.explained_variance_ratio_.sum()),
      "factor_count":fa_n,"summaries":summaries,"points":points,"numeric_columns":numeric_cols,
      "preview":display.head(8).to_dict("records"),"preview_columns":display.columns.tolist()}
