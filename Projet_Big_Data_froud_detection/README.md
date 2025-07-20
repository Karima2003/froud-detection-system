# froud-detection-system
# 🚨 Fraud Detection System using Big Data Technologies

This project is a real-time fraud detection system built with **Apache Kafka**, **Apache Spark**, and **Machine Learning**, designed to identify fraudulent transactions from a live stream of data.

## 📁 Project Structure

Projet_Big_Data_froud_detection/
├── data/ # Sample transaction data
├── kafka/ # Kafka configuration & producers
├── spark/ # Spark Streaming scripts
├── models/ # ML models (e.g., RandomForest)
├── requirements.txt # Python dependencies
└── README.md # This file

## 🚀 Features

- 🧠 Machine Learning model to detect fraudulent transactions
- ⚡ Real-time stream processing using Spark Structured Streaming
- 📡 Integration with Kafka for data ingestion
- 📊 Displays recommended actions or flags on suspicious data
- 🧪 Handles class imbalance using weights or sampling
- 🛡️ Scalable and modular architecture

## 🛠️ Technologies Used

| Tool         | Purpose                            |
|--------------|-------------------------------------|
| Python       | ML training and orchestration       |
| Apache Kafka | Data ingestion (stream of events)   |
| Apache Spark | Real-time stream processing         |
| Scikit-learn | Machine Learning classification     |
| Docker       | Containerization                    |

## 🧠 Machine Learning Model

The fraud detection model is trained using:
- **Random Forest Classifier**
- Takes into account **class imbalance** by applying weights
- Features engineered from transaction metadata

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/Karima2003/froud-detection-system.git
cd froud-detection-system
