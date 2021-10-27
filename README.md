This repository supports a manuscript submission to ESSD where we used a Bayesian Neural Network [framework](https://proceedings.neurips.cc/paper/2020/file/0d5501edb21a59a43435efa67f200828-Paper.pdf) to ensemble Chemistry climate models from [CCMI](https://igacproject.org/activities/CCMI)

### If you are here for the data
- ML training data preprocessed and ready to go is [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5603750.svg)](https://doi.org/10.5281/zenodo.5603750). This data has the same extrapolation and interpolation testing that we used in the manuscript. This data should be combined with code in Training/preprocess_data.py
- Infilled zonal mean ozone (vertically resolved): [zmo3_BNNOz.nc](https://github.com/mattramos/VertOzone-BNN/zmo3_BNNOz.nc) (Also versioned [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5603750.svg)](https://doi.org/10.5281/zenodo.5603750))

### If you are here for a template of the BNN
- Look [here for a toy example](https://github.com/mattramos/Toy-bayesian-neural-network-ensemble) also in binder [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mattramos/Toy-bayesian-neural-network-ensemble/master?filepath=toy_dataset_example.ipynb)
- Look here for a blank repository (WORK IN PROGRESS - send me an email)

### If you want to know what we did
The below flow chart shows the process with links to show sources of data and which scripts we used. As much of the data we used is external we have not provided it here. This includes the other ozone datasets.
Other code sources  
- Dynamical linear modelling for ozone trend analysis from [Alsing et al., (2019)](https://github.com/justinalsing/dlmmc). Datasets used as regressors are found within the DLM documentation.  

Other datasets
- Our training dataset [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5603750.svg)](https://doi.org/10.5281/zenodo.5603750)
- [Chemistry climate model data](https://data.ceda.ac.uk/badc/wcrp-ccmi/data/CCMI-1/output)
- [Bodeker Scientific](http://www.bodekerscientific.com/data/monthly-mean-global-vertically-resolved-ozone)
- [SWOOSH](https://csl.noaa.gov/groups/csl8/swoosh/)
- [SBUV](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2001JD000853)
- [BASIC](https://data.mendeley.com/datasets/2mgx2xzzpk/3)

<pre>
&nbsp;&nbsp;&nbsp;┌──────────────┐&nbsp;&nbsp;┌─────────────────┐&nbsp;&nbsp;  
&nbsp;&nbsp;&nbsp;│Model&nbsp;data&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;│Observation&nbsp;data&nbsp;│&nbsp;&nbsp;
   │----------    │  │---------------- │
   └──────┬───────┘  └──────┬──────────┘
          │                 │
          ▼                 ▼
┌──────────────────────────────────────────┐
│ Preprocessing                            │
│ ----------------                         │
│ preprocessing_training_dataV2.1.ipynb    │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│ Training                                 │
│ ------------------                       │
│ BNN model                                │
│ ┌─►BNN.py                                │
│ │┤►multi_train.py                        │
│ │┤►preprocess_data.py                    │
│ │┤►utils.py                              │
│ └─►dispatch_script.ipynb                 │
│ checking_priors.ipynb                    │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│ Postprocessing  (testing / validating)   │
│ -------------------------------------    │
│ load_and_save_raw_output.ipynb           │
│ constructing_zmo3_output.ipynb           │
└───────┬──────────────────────┬───────────┘
        │                      │
        ▼                      ▼
┌────────────────┐   ┌─────────────────────┐
│ Plotting       │   │ DLM                 │
│ --------       │   │ ---                 │
└────────────────┘   │ Code from Alsing    │
                     │ Extra datasets      │
                     └─────────────────────┘
                     
</pre>
