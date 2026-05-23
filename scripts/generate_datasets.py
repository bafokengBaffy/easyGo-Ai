from pathlib import Path
import csv
import random

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

def generate_rider_dataset(path: Path, n: int = 2000):
    header = [
        "rider_id",
        "age",
        "signup_days",
        "trips_last_30d",
        "avg_trip_distance_km",
        "avg_rating_given",
        "churned",
        "estimated_ltv",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(1, n + 1):
            age = random.choice([None] * 5 + list(range(18, 70)))
            signup_days = random.randint(1, 2000)
            trips = max(0, int(random.gauss(5, 8)))
            dist = round(abs(random.gauss(5.0, 3.0)), 2)
            rating = round(random.uniform(3.0, 5.0), 2)
            churned = 1 if random.random() < 0.08 + (trips < 1) * 0.2 else 0
            ltv = round(max(0.0, random.gauss(120.0 + trips * 5, 50.0)), 2)
            writer.writerow([f"r{i:06}", age, signup_days, trips, dist, rating, churned, ltv])

def generate_driver_dataset(path: Path, n: int = 2000):
    header = [
        "driver_id",
        "age",
        "years_experience",
        "acceptance_rate",
        "avg_response_time_s",
        "trips_last_30d",
        "avg_ride_time_min",
        "eta_error_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(1, n + 1):
            age = random.choice([None] * 3 + list(range(21, 65)))
            yrs = random.randint(0, 25)
            accept = round(min(1.0, max(0.0, random.gauss(0.85, 0.12))), 3)
            resp = round(abs(random.gauss(20, 15)), 1)
            trips = max(0, int(random.gauss(30, 25)))
            ride_time = round(abs(random.gauss(18, 10)), 2)
            eta_err = round(random.gauss(60, 120), 1)
            writer.writerow([f"d{i:06}", age, yrs, accept, resp, trips, ride_time, eta_err])

def main():
    rider_path = RAW / "rider_dataset.csv"
    driver_path = RAW / "driver_dataset.csv"
    print(f"Generating rider dataset -> {rider_path}")
    generate_rider_dataset(rider_path, n=2200)
    print(f"Generating driver dataset -> {driver_path}")
    generate_driver_dataset(driver_path, n=2100)
    print("Datasets generated (raw CSVs).")

if __name__ == '__main__':
    main()
