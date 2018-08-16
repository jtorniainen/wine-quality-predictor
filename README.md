# wine-quality-predictor
Predicting them wines.

Contents of the repository:
- wine_automl.py: script for running the auto-sklearn analysis
- wine_linreg.py: script for running the linear regression analysis
- winequality-white.csv: the dataset
- winequality.names: information about the dataset
- automl.log: log file of the auto-sklearn run (with preprocessing)
- automl_no_pp.log: log file of the auto-sklearn run (without preprocessing)
- visualize_automl_pipes.py: visualizer for different analysis methods tried by auto-sklearn
- parse_automl_logs.py: parser for auto-sklearn logs
- conf_matrix.py: visualizer for the confusion matrices
- wine_data.py: preliminary analysis of the dataset (distributions and collinearity)
- automl_results.py: visualize the results of the auto-sklearn analysis
