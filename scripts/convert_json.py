import csv
import json
import os

# Get absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "dataset", "dataset_training.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "dataset", "dataset_training_bert.csv")

def convert_to_json(response_text):
    """
    Konversi response berbasis delimiter "|" menjadi JSON.
    Atur rule sesuai kebutuhanmu.
    """

    # Jika mengandung "|" berarti LIST
    if "|" in response_text:
        parts = [p.strip() for p in response_text.split("|") if p.strip()]

        return json.dumps({
            "type": "list",
            "title": parts[0],      # ambil kalimat utama
            "items": parts[1:],     # sisanya jadi list
        }, ensure_ascii=False)

    # Jika tidak ada "|" → TEXT biasa
    return json.dumps({
        "type": "text",
        "body": response_text.strip()
    }, ensure_ascii=False)


def process_csv():
    rows = []

    with open(INPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames

        for row in reader:
            # Convert hanya kolom response
            row["response"] = convert_to_json(row["response"])
            rows.append(row)

    # Output
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print("✔️ Dataset berhasil diproses dan disimpan ke", OUTPUT_CSV)


if __name__ == "__main__":
    process_csv()
