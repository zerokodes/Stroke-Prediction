# Stroke Prediction Model

## Overview
This project focuses on creating a predictive model to determine the likelihood of a patient having a stroke based on their personal and medical information. The dataset used is sourced from [Kaggle](https://www.kaggle.com/datasets). The model leverages machine learning techniques to analyze factors such as age, gender, various health conditions, and smoking status.

## Features
- Predicts stroke risk based on input parameters.
- Uses a Kaggle dataset with comprehensive patient data.
- Built with Python and popular data science libraries such as Pandas, Scikit-learn, and Matplotlib.

## Dataset
The dataset contains rows representing individual patients with the following fields:
- **Gender**: Male, Female, or Other
- **Age**: Patient's age
- **Hypertension**: 0 (no) or 1 (yes)
- **Heart Disease**: 0 (no) or 1 (yes)
- **Marital Status**: Married or Single
- **Work Type**: Type of work (Private, Government, etc.)
- **Residence Type**: Urban or Rural
- **Average Glucose Level**: Average blood glucose level
- **BMI**: Body Mass Index
- **Smoking Status**: Current smoker, Former smoker, or Never smoked
- **Stroke**: Target variable (0 = No stroke, 1 = Stroke)

The dataset can be downloaded from the [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets).

---

## Requirements
Ensure you have the following installed:
- Python (>= 3.8)
- pip

### Python Libraries
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the required libraries using the command:
```bash
pip install -r requirements.txt
```

---

## Project Structure
```plaintext
stroke-prediction/
|— data/                 # Directory to store the dataset
|— notebooks/            # Jupyter notebooks for exploration
|— deployment/               # local and web deployment
|— README.md            # Project documentation
```

---

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/stroke-prediction.git
   cd stroke-prediction
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets).
   - Place the dataset file (e.g., `stroke-data.csv`) in the `data/` directory.

4. **Run Exploratory Data Analysis (EDA)**
   Open and run the Jupyter notebook in the `notebooks/` folder for initial data analysis and visualization.

   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```

5. **Train the Model**
   Run the training script to preprocess data and train the model.
   ```bash
   python src/train.py
   ```

6. **Make Predictions**
   Use the prediction script to predict stroke likelihood for new patient data.
   ```bash
   python src/predict.py --input path_to_input_file.csv
   ```

---

## Model Workflow
1. **Data Preprocessing**:
   - Handling missing values
   - Encoding categorical variables
   - Scaling numerical features

2. **Model Selection**:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting

3. **Evaluation Metrics**:
   - Accuracy
   - Precision
   - Recall
   - F1-Score

4. **Prediction**:
   - The trained model outputs the probability of stroke occurrence based on input features.

---

## How to Contribute
We welcome contributions! Follow these steps:
1. Fork the repository.
2. Create a new branch for your feature.
3. Commit your changes.
4. Push to the branch.
5. Open a pull request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Acknowledgements
- Dataset: [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets)
- Python Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn


