# DA-chestxray
chest x-ray image classifiaction domain adaptation

the main driver code is located in the /Data and codes/cxrnoobcoder directory, and the file name is main_cli.py
this file is meant to be accessed through command line interface (cli) and the arguments help is available in the code itself
use ```python main_cli.py -h```

example of executing domain adaptation from NIH to MIMIC:
1. train on NIH set

   ```
   python main_cli.py --run_id DA_1 --data_path ../nih-kaggle/images --csv_path ../nih-kaggle
   --csv_name 'NIH_Original label_pp_use this.csv' --model densenet --bs 64 --n_epochs 20
   --loss_func BCE --weight_decay 0.0001
   ```
   
   the training will produce a model file according to your run_id, in this case: <b>model_DA_1.pth</b>
   you can see the metrics of the training

2. test on MIMIC set: Remember to include -t argument for testing only (no training)
   ```
   python main_cli.py --run_id DA_1 --data_path ../MIMIC-CXR-2.0 --csv_path ../MIMIC-CXR-2.0 --csv_name all_mimic_labels.csv
   --model densenet --bs 64 --loss_func BCE --weight_decay 0.0001 -t
   ```

Results of some trainings (models, command line output, loss curve, metrics) are in /cxrnoobcoder/experiments.
the file /cxrnoobcoder/experiments/test_results_all_summary.csv contains all results' metric.


   
