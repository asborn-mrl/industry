# 🏭 Industrial Safety Monitoring System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An **XGBoost-based machine learning system** for predicting worker accident probability in industrial environments. Developed for IEEE Conference Paper publication.

![App Screenshot](assets/screenshot.png)

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 65.1% |
| **Precision** | 69.2% |
| **Recall** | 85.2% |
| **F1-Score** | 0.764 |
| **ROC-AUC** | 0.616 |

> **Note**: The high recall (85.2%) ensures the model effectively identifies potential accident scenarios, which is critical for safety applications.

---

## 🚀 Features

- **🔮 Single Prediction**: Real-time risk assessment for individual workers
- **📁 Batch Analysis**: Upload CSV for bulk predictions
- **📈 Interactive Dashboard**: Visualize model performance and feature importance
- **🎯 Risk Categorization**: LOW, MEDIUM, HIGH, CRITICAL levels
- **💡 Actionable Recommendations**: Safety suggestions based on risk factors
- **📊 Beautiful Visualizations**: Gauge charts, confusion matrix, feature importance

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/industrial-safety-monitoring.git
cd industrial-safety-monitoring

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
industrial-safety-monitoring/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore               # Git ignore file
│
├── models/                   # Trained ML models
│   └── xgboost_model.pkl    # XGBoost classifier
│
├── data/                     # Dataset files
│   └── sample_data.csv      # Sample input data
│
└── assets/                   # Images and resources
    └── screenshot.png       # App screenshot
```

---

## 📋 Input Features

The model analyzes **21 features** across 6 categories:

| Category | Features |
|----------|----------|
| **Worker Demographics** | Age, Experience, Department, Shift |
| **Environmental** | Temperature, Humidity, Noise Level, Lighting |
| **Equipment** | Age, Maintenance Days, Condition |
| **Work Conditions** | Hours Worked, Overtime, PPE, Fatigue |
| **Training** | Training Status, Days Since Training |
| **Historical** | Previous Incidents, Near Misses |

---

## 🎯 Risk Levels

| Level | Probability | Action Required |
|-------|-------------|-----------------|
| 🟢 **LOW** | 0-25% | Routine monitoring |
| 🟡 **MEDIUM** | 25-50% | Enhanced supervision |
| 🟠 **HIGH** | 50-75% | Immediate review |
| 🔴 **CRITICAL** | 75-100% | Work stoppage |

---

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub repository
4. Select `app.py` as main file
5. Click Deploy!

### Deploy to Other Platforms

<details>
<summary>Heroku</summary>

```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```
</details>

<details>
<summary>Docker</summary>

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```
</details>

---

## 📖 Usage

### Single Prediction

1. Navigate to "📊 Single Prediction"
2. Enter worker and environmental details
3. Click "Calculate Risk Assessment"
4. View probability, risk level, and recommendations

### Batch Analysis

1. Navigate to "📁 Batch Analysis"
2. Upload a CSV file with worker data
3. Click "Run Batch Analysis"
4. Download results with predictions

---

## 🔬 Model Details

### Algorithm: XGBoost

| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| max_depth | 6 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| objective | binary:logistic |

### Top 5 Important Features

1. **noise_level** (0.0895)
2. **days_since_training** (0.0864)
3. **experience_years** (0.0816)
4. **hours_worked_today** (0.0728)
5. **supervision_ratio** (0.0710)

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@inproceedings{industrial_safety_2024,
  title={Industrial Safety Monitoring System Using XGBoost Decision Tree Model
         for Worker Accident Probability Prediction},
  author={Kumar, Rajesh and Sharma, Priya and Patel, Amit and Krishnan, Deepa},
  booktitle={IEEE Conference},
  year={2024}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

<p align="center">
  Made with ❤️ for Industrial Safety
</p>
