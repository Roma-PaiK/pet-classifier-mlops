import argparse
import os
import csv
import requests

def iter_labeled_images(root):
    """Yield (true_label, path) for images in root/Cat/ and root/Dog/."""
    for label in ["Cat", "Dog"]:
        folder = os.path.join(root, label)
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                yield label, os.path.join(folder, fn)

def _mime_for(path):
    ext = os.path.splitext(path)[1].lower()
    return "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

def iter_flat_images(root):
    """Yield (None, path) for images directly in root (no labels)."""
    if not os.path.isdir(root):
        return
    for fn in sorted(os.listdir(root)):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            yield None, os.path.join(root, fn)

def load_labels_csv(path):
    """Load {filename: "Cat"|"Dog"} from CSV with columns file, true_label (or label)."""
    labels = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            fn = row.get("file") or row.get("filename")
            lbl = (row.get("true_label") or row.get("label") or "").strip()
            if fn and lbl and lbl in ("Cat", "Dog"):
                labels[fn] = lbl
    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--data", default="eval_data")
    ap.add_argument("--labels", default=None, help="CSV with file, true_label (or use data/labels.csv)")
    ap.add_argument("--out", default="eval_results.csv")
    args = ap.parse_args()

    rows = []
    total = 0
    correct = 0

    # Prefer labeled layout (Cat/, Dog/); else flat folder, optionally with labels CSV
    labeled = list(iter_labeled_images(args.data))
    if labeled:
        it = labeled
    else:
        flat = list(iter_flat_images(args.data))
        labels_path = args.labels or os.path.join(args.data, "labels.csv")
        labels_map = load_labels_csv(labels_path) if os.path.isfile(labels_path) else {}
        # Pair each image with its label from CSV if present
        it = [(labels_map.get(os.path.basename(p), None), p) for _, p in flat]

    for true_label, path in it:
        with open(path, "rb") as f:
            name = os.path.basename(path)
            files = {"file": (name, f, _mime_for(path))}
            resp = requests.post(f"{args.base_url}/predict", files=files, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        pred = payload.get("prediction")
        conf = payload.get("confidence")
        ok = (pred == true_label) if true_label is not None else ""
        total += 1
        if true_label is not None:
            correct += int(ok)
        rows.append([os.path.basename(path), true_label or "", pred, conf, ok])

    if total:
        if any(r[1] for r in rows):  # had labels
            acc = correct / total
            print(f"Post-deploy accuracy over {total} images: {acc:.4f}")
        else:
            print(f"Post-deploy predictions for {total} images (no labels, accuracy N/A)")
    else:
        print("No images found. Use root/Cat/ and root/Dog/ for accuracy, or put images in data root.")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "true_label", "pred_label", "confidence", "correct"])
        w.writerows(rows)

if __name__ == "__main__":
    main()
