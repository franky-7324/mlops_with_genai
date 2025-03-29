import kagglehub

# Download latest version
path = kagglehub.dataset_download("khushikyad001/indian-traffic-violation", local_dir="/mnt/d/Github2025_v2/prices-predictor-system/indian-traffic-violation/data")

print("Path to dataset files:", path)