# Binary Classification with Neural Networks on the Census Income Dataset

This workshop builds a binary classification model using **PyTorch** to predict whether an individual earns more than **$50,000 annually**.

## Dataset

The supplied Census Income Dataset contains **30,000 entries**.

The original dataset link provided for the workshop is:
https://drive.google.com/file/d/1ay5vOv2YiOwjKIWnXFT6sptW0gqew0oa/view?usp=sharing

A copy of `income.csv` is included in `data/` so the notebook can be run directly from this repository.

## Model Requirements

The implementation follows the workshop specification:

- 25,000 training samples
- 5,000 testing samples
- Categorical feature embeddings
- Batch normalization for continuous features
- One hidden layer with 50 neurons
- Dropout `p=0.4`
- `CrossEntropyLoss`
- Adam optimizer
- Learning rate `0.001`
- 300 training epochs
- Test loss and accuracy evaluation
- Optional new-data prediction function

## Project Structure

```text
census-income-workshop/
├── data/
│   └── income.csv
├── notebooks/
│   └── census_income_workshop.ipynb
├── requirements.txt
└── README.md
```

## Features Used

### Categorical
- `sex`
- `education`
- `marital-status`
- `workclass`
- `occupation`

### Continuous
- `age`
- `education-num`
- `hours-per-week`

### Target
- `label`

The `income` text column is not used as an input because it directly represents the target class.

## Installation

Install Python 3 and then install the dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/census_income_workshop.ipynb
```

## Results

Using the required configuration and random seed:

- **Test Loss:** approximately `0.2670`
- **Test Accuracy:** approximately `88.02%`


