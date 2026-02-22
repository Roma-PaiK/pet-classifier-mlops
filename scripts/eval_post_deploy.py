import argparse
import os
import csv
import requests

def iter_images(root):
    for label in ["Cat", "Dog"]:
        folder = os.path.join(root, label)
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                yield label, os.path.join(folder, fn)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--data", default="eval_data")
    ap.add_argument("--out", default="eval_results.csv")
    args = ap.parse_args()

    rows = []
    total = 0
    correct = 0

    for true_label, path in iter_images(args.data):
        with open(path, "rb") as f:
            resp = requests.post(f"{args.base_url}/predict", files={"file": f}, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        pred = payload.get("prediction")
        conf = payload.get("confidence")
        ok = (pred == true_label)
        total += 1
        correct += int(ok)
        rows.append([os.path.basename(path), true_label, pred, conf, ok])

    acc = (correct / total) if total else 0.0
    print(f"Post-deploy accuracy over {total} images: {acc:.4f}")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "true_label", "pred_label", "confidence", "correct"])
        w.writerows(rows)

if __name__ == "__main__":
    main()
