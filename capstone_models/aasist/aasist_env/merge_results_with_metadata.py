import pandas as pd
from pathlib import Path

# Define paths
metadata_file = Path("./new_evaluation_metadata.csv")


results_file = Path("/Users/alecnaidoo/Capstone_2025/capstone_models/aasist/aasist_env/exp_result/NB_NB_AASIST_ep75_bs24/mozilla_evaluation_results6.csv")
# results_file_L = Path("/Users/alecnaidoo/Capstone_2025/capstone_models/aasist/aasist_env/exp_result/LA_AASIST-L_ep100_bs24/mozilla_evaluation_results5.csv")
output_file = Path("./final_results_NB_AASIST_epoch_71_0.018.csv")

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