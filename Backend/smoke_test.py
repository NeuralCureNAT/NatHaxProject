from app import app

def test_smoke():
    with app.test_client() as c:
        assert c.get("/api/health").status_code == 200
        assert c.get("/api/eeg/history?limit=1").status_code == 200
        r = c.get("/api/eeg/current")
        assert r.status_code in (200, 204)  # 204 when no current sample

if __name__ == "__main__":
    test_smoke()
    print("✅ smoke ok")
