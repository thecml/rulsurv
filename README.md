# rulsurv
Code for: "RULSurv: A probabilistic survival-based method for early censoring-aware prediction of remaining useful life in ball bearings (2025)"<br />

Preprint: https://arxiv.org/abs/2405.01614 **(Under review)**

Requirements
----------------------
To run the models, please refer to [Requirements.txt](https://github.com/thecml/rulsurv/blob/main/requirements.txt).
The code was tested in a virtual environment with `Python 3.9`.

Training and prediction
--------
- Download the XJTU-SY dataset from [Dropbox](https://www.dropbox.com/scl/fi/ffneg1y7122jp5axlebpv/rulsurv_data.zip?rlkey=vd4yszi8vexhl7yaib4yfue53&st=gz1knl1b&dl=0)
- Unzip the data inside the main folder to create /data
- Make directories `mkdir results` and `mkdir plots`
- Make the timeseries dataset by running `python src/make_dataset.py`
- Run cross-validation by running `python src/run_cross_validation.py`
- Predict ISDs by running `python src/predict_isd_curves.py`
- Refer to `config.py` for model configuration and paths.

Citation
--------
If you find this paper useful in your work, please consider citing it:
 
```
@article{lillelund_rulsurv_2025,
  title={RULSurv: A probabilistic survival-based method for early censoring-aware prediction of remaining useful life in ball bearings}, 
  author={Christian Marius Lillelund and Fernando Pannullo and Morten Opprud Jakobsen and Manuel Morante and Christian Fischer Pedersen},
  journal={preprint, arXiv:2405.01614},
  year={2025},
}
```
