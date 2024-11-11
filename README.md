
# Battery State Estimation Projects (SOX)

This repository serves as a personal self-study initiative aimed at deepening my understanding of lithium-ion battery management systems. Through this project, I aim to gain practical experience in exploring SOC and SOH estimation techniques while identifying areas for self-improvement in my own knowledge and skills. It is designed more so as a sandbox for me to develop, test and implement state estimation techniques for various sample li-ion batteries.

## Objectives (aimed at addressing areas for self-improvement)

- **Learn**: Gain knowledge about simple SOX estimatino techniques, and branch towards more sophisticated SOC estimation techniques, such as Extended Kalman Filters or machine learning-based approaches.
- **Account for Real Life**: Explore how to work with real-world battery datasets with variable quality to validate models and improve accuracy.
- **KPIs**: Understand key performance indicators for BMS, including efficiency and reliability metrics.
- **Avoid obsolence**: Testing either recent software solutions or published theories from papers to calibrate understanding to latest global view.

## Project Structure

This project is organized into several directories, each serving a specific purpose:
```
battery_estimation/
├── data/ # Directory for storing datasets (if applicable)
├── models/ # Contains modules for SOC and SOH estimation
├── utils/ # Utility functions for data generation and processing
├── notebooks/ # Jupyter notebooks for interactive exploration
│ └── battery_estimation.ipynb # Main notebook for testing and visualization
├── README.md # Project overview and documentation
└── requirements.txt # List of dependencies required to run the project
```
## Getting Started

### Prerequisites

To run this project, you will need Python installed on your machine along with the required libraries. You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Authors

- [@javaidb](https://www.github.com/javaidb)


## Acknowledgements

 - [Mendeley EIS Dataset 1](https://data.mendeley.com/datasets/n78tkm784n/1)
 - [Impedance.py + Sample Dataset](https://impedancepy.readthedocs.io/en/latest/getting-started.html)

Models built on TensorFlow and modeled on datasets [LG 18650HG2](https://data.mendeley.com/datasets/cp3473x7xv/3) and  [Panasonic 18650PF](https://data.mendeley.com/datasets/wykht8y7tg/1).

## Badges

[![IBM License](https://img.shields.io/badge/Certificate_ML-IBM-green.svg)](https://www.credly.com/badges/6d82b78c-cade-4a4c-94cb-b7f89e142350/public_url)