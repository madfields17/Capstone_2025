import pandas as pd
from pathlib import Path

# Define paths
metadata_file = Path("../../datasets/evaluation-data/evaluation-metadata.csv")
results_file = Path("../finetuned-model/tssd-evaluation-results.csv")
output_file = Path("../finetuned-model/final-results-tssd.csv")

# Load metadata
metadata = pd.read_csv(metadata_file)

# Load model predictions
results = pd.read_csv(results_file)

# Extract only the filename from the `wav_path` column
results["wav_filename"] = results["wav_path"].apply(lambda x: Path(x).name)

# Ensure filenames match by converting .mp3 names to .wav in metadata
metadata["wav_filename"] = metadata["file_name"].str.replace(".mp3", ".wav", regex=False)

# Merge metadata with results on the cleaned WAV file name
merged_df = metadata.merge(results, on="wav_filename", how="left")

# Save the final merged dataset
merged_df.to_csv(output_file, index=False)
print(f"✅ Merged results saved to {output_file}")