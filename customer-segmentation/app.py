from flask import Flask, render_template, request, flash
from src.segmentation import run_analysis, load_sample_data
import os, pandas as pd

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            k = int(request.form.get("clusters", 4))
            if not 2 <= k <= 8:
                raise ValueError("Choose between 2 and 8 clusters.")
            upload = request.files.get("dataset")
            if upload and upload.filename:
                if not upload.filename.lower().endswith(".csv"):
                    raise ValueError("Please upload a CSV file.")
                df = pd.read_csv(upload)
                source = upload.filename
            else:
                df = load_sample_data()
                source = "Built-in synthetic demonstration dataset"
            result = run_analysis(df, k)
            result["source"] = source
        except Exception as exc:
            flash(str(exc), "error")
    return render_template("index.html", result=result)

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
