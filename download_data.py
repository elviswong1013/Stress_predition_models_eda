import os
import glob
import shutil
import kagglehub

def download_dataset():
    path = kagglehub.dataset_download("amar5693/screen-time-sleep-and-stress-analysis-dataset")
    print("Path to dataset files:", path)
    # Find CSV files in the downloaded directory
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        # search recursively
        csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the downloaded dataset.")
    # Prefer a file that likely contains the data
    target_csv = None
    for f in csv_files:
        name = os.path.basename(f).lower()
        if "stress" in name or "screen" in name or "dataset" in name:
            target_csv = f
            break
    if target_csv is None:
        target_csv = csv_files[0]
    dest = os.path.join(os.getcwd(), "dataset.csv")
    shutil.copyfile(target_csv, dest)
    print(f"Copied {target_csv} to {dest}")

if __name__ == "__main__":
    download_dataset()
