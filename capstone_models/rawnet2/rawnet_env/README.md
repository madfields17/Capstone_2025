## RawNet2
This is modified code of the rawnet2 model. Reference original repo here: https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2

Incomplete:
- Point training script to training dataset.
- Fine tune model to training data using GPU resources. 

Complete:
- Evaluate it against mozilla_evaluation_wav folder which contains our final evaluation set. Run `test_rawnet2.py` which is modified to look at mozilla_evaluation_wav for the final probability predictions.
    - This returns the probability of it being a "bonafide" sample. 
- Use the ipynb notebook `capstone_models/model_analysis.ipynb` for more information on results & comparison analysis. 

Please let know if there are any issues!