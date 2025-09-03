# (MIDAG-SPCN) Identity Unlocks the Prism: A Graph-Informed Automatic Spectral Convolution Framework for Multimedia Time Series Forecasting

### Datasets

>> "raw_files.zip" contains 14 multivariate time series datasets while "TaTS-main/data" displays 8 multimodal time series datasets.

## running programme

### Running style for "multivariate" scenario:


> >(1) cd Time-Series-Library-main.

> >(2) run: nohup bash scripts/long_term_forecast/ECL_script/MIDAG_SPCN.sh > midag_spcn_ecl.log 2>&1 & **(please use your own root path (e.g., --root_path "your_patch" + /electricity/))**.

> >(3) The results are in the corresponding midag_spcn_ecl.log file.


###  Running style for "multimodal" scenario:

> >(1) cd TaTS-main.

> >(2)check your hypermeters based on fold "hypermeters_references" when facing different datasets.

> >(3) run: nohup bash scripts/main_forecast_midag_spcn.sh > midag_spcn.log 2>&1 &

> >(4) The results are in the corresponding midag_spcn.log file.