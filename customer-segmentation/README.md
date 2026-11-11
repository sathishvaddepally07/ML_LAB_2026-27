# CustomerLens — Customer Segmentation Studio

A Flask + scikit-learn mini-project demonstrating four algorithms/methods from the ML syllabus:

1. **K-Means** — partitions customers into clusters.
2. **Gaussian Mixture Model (GMM)** — probabilistic clustering.
3. **PCA** — dimensionality reduction and 2D visualisation.
4. **Factor Analysis** — latent-factor representation.

**Academic note:** PCA and Factor Analysis are dimensionality-reduction methods, not clustering algorithms. They are included as syllabus methods for representing and exploring customer patterns. If your instructor requires four distinct clustering algorithms, confirm whether this combination is acceptable.

## Dataset
`data/sample_customers.csv` is a **synthetic demonstration dataset** with 200 generated records. It is not real customer research. The app accepts an uploaded CSV too. Never present the synthetic data as real collected data.

## Run locally
Python 3.10–3.12 recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

Windows activation: `.venv\\Scripts\\activate`

## Upload your own data
Upload a CSV with at least 5 rows. Numeric columns are median-imputed and scaled; categorical columns are imputed and one-hot encoded. ID/name columns are removed automatically.

## Deploy on Render
- Runtime: Python 3
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`

## Limitations
- Synthetic sample data is only for demonstration.
- Cluster numbers are arbitrary labels, not quality rankings.
- Silhouette score is a clustering diagnostic, not prediction accuracy.
- GMM and K-Means may assign customers differently.
