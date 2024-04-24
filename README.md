# rulsurv
Code for: A probabilistic estimation of remaining useful life from censored time-to-event data (2024)<br />

Requirements
----------------------
To run the models, please refer to [Requirements.txt](https://github.com/thecml/rulsurv/blob/main/requirements.txt).
The code was tested in a virtual environment with `Python 3.9`.

Training and prediction
--------
- Download the XJTU-SY dataset from [Dropbox](https://www.dropbox.com/scl/fi/bpfoygq7xe1yjvl0w6esn/data.zip?rlkey=ao00hr46to3u6iy9patvlmcyr&st=ndy6cwc6&dl=0)
- Unzip the data inside the main folder to create /data
- Make directories `mkdir results` and `mkdir plots`
- Make the timeseries dataset by running `python src/make_dataset.py`
- Run cross-validation by running `python src/run_cross_validation.py`
- Predict ISDs by running `python src/plot_isd_curves.py`
- Refer to `config.py` for model configuration and paths.

Citation
--------
TBA
