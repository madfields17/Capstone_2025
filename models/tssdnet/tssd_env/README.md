## TTSD
This is modified code of the TTSD-net model. Reference original repo here: https://github.com/ghua-ac/end-to-end-synthetic-speech-detection

Incomplete:

Point training script to training dataset.
Fine tune model to training data using GPU resources.

Complete:

Evaluate it against mozilla_evaluation_wav folder (did not include but can if necessary - see other model folders) which contains our final evaluation set. Run test_tssdnet.py which is modified to look at mozilla_evaluation_wav for the final probability predictions.
This returns the probability of it being a "bonafide" sample.
Use the ipynb notebook capstone_models/model_analysis.ipynb for more information on results & comparison analysis.

Please let know if there are problems with the code or if you run into any issues!