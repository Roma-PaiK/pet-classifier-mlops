import sys
import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

def main():
    r = requests.get(f"{BASE}/health", timeout=10)
    r.raise_for_status()
    print("health:", r.json())

    m = requests.get(f"{BASE}/metrics", timeout=10)
    m.raise_for_status()
    print("metrics: OK")

if __name__ == "__main__":
    main()
