# Metritis Risk Classification

This project focuses on developing machine learning models to classify and predict metritis risk in dairy cows. The system uses various machine learning algorithms to analyze and predict the risk factors associated with metritis.

## Setup Instructions

1. Create a conda environment:
```bash
conda create -n metritis python=3.9
conda activate metritis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

- `data/`: Contains the dataset files
- `run_experiment.py`: Main script for running experiments
- `notebook_metritis_risk_classification_v2.ipynb`: Jupyter notebook for analysis and visualization
- `requirements.txt`: List of Python package dependencies

## Usage

1. Ensure your data is placed in the `data/` directory
2. Run the experiment script:
```bash
python run_experiment.py
```

3. For interactive analysis, open the Jupyter notebook:
```bash
jupyter notebook notebook_metritis_risk_classification_v2.ipynb
```

## License

This project is licensed under the terms specified in the LICENSE file.