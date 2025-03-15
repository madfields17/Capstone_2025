# AASIST - Deepfake Audio Detection

This repository provides an implementation of the **AASIST model** for detecting deepfake (spoofed) audio. The setup allows evaluation on custom datasets, including a curated version of **Mozilla Common Voice** for demographic bias analysis.

## **1. Setting Up the Poetry Environment**
We use **Poetry** for dependency management. Follow these steps to install and set up the environment:

### **1.1 Install Poetry (if not installed)**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
or use:

```bash 
pip install poetry
```


### **1.3 Install Dependencies**
```bash
poetry install
```

---

## **2. Preparing the Evaluation Dataset**
The model expects **WAV files** sampled at **16 kHz** for evaluation. Convert your MP3 files using `convert_mp3_to_wav.py`.

### **2.1 Move Your MP3 Files to the `mozilla_evaluation_set/` Folder**
```bash
mkdir -p mozilla_evaluation_set
mv /path/to/your/mp3/files/*.mp3 mozilla_evaluation_set/
```

### **2.2 Convert MP3 Files to WAV (16 kHz)**
Run the following script to convert all MP3 files:
```bash
poetry run python convert_mp3_to_wav.py
```
✅ **Converted WAV files will be stored in `mozilla_evaluation_wav/`**

---

## **3. Running Evaluation**
Once the dataset is ready, run the evaluation script:

```bash
poetry run python main.py --eval --config ./config/AASIST.conf
```

This will:
1. Load the **AASIST** deepfake detection model.
2. Evaluate **Mozilla Common Voice** WAV files.
3. Save results in:
   ```
   exp_result/LA_AASIST_ep100_bs24/mozilla_evaluation_results.csv
   ```

### **3.1 View Evaluation Results**
```bash
cat exp_result/LA_AASIST_ep100_bs24/mozilla_evaluation_results.csv
```
The CSV will contain:
```
wav_path,prediction_score
mozilla_evaluation_wav/sample1.wav,0.92
mozilla_evaluation_wav/sample2.wav,0.03
```
- **Values close to 1** → More likely **spoofed (fake)**
- **Values close to 0** → More likely **real**

### **3.2 Merge Results with Metadata for Regional Groupings**
Run the following script to convert all MP3 files:
```bash
cd ..
poetry run python aasist_env/merge_results_with_metadata.py
```
---

## **4. Troubleshooting**
### **4.1 No GPU?**
If you get:
```
ValueError: GPU not detected!
```
Modify `main.py` to **allow CPU execution**:
```python
# Comment out the GPU check:
# if device == "cpu":
#     raise ValueError("GPU not detected!")

print("⚠ Warning: Running on CPU. This may be significantly slower.")
```

### **4.2 Poetry Not Found?**
Ensure Poetry is in your PATH:
```bash
export PATH="$HOME/.poetry/bin:$PATH"
```

---