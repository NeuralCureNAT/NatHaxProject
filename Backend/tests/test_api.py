import os, csv, tempfile

# Point the app at a throwaway CSV BEFORE importing app
tmp_csv = os.path.join(tempfile.gettempdir(), "neuralcare_test.csv")
os.environ["NEURALCARE_PROCESSED_CSV"] = tmp_csv

from app import app  # noqa: E402

def setup_module(_):
    # create a tiny CSV with one row
    with open(tmp_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "alpha", "beta", "theta"])
        w.writeheader()
        w.writerow({"ts": "100", "alpha": "0.1", "beta": "0.2", "theta": "0.3"})

def test_health():
    with app.test_client() as c:
        r = c.get("/api/health")
        assert r.status_code == 200
        data = r.get_json()
        # Your health payload includes at least a status/time
        assert isinstance(data, dict)
        assert "status" in data

def test_history_limit():
    with app.test_client() as c:
        r = c.get("/api/eeg/history?limit=1")
        assert r.status_code == 200
        rows = r.get_json()
        assert isinstance(rows, list)
        assert len(rows) <= 1

def test_current_endpoint_exists():
    with app.test_client() as c:
        r = c.get("/api/eeg/current")
        assert r.status_code in (200, 204)