
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
    battery-state-estimation/
    ├── datasets/                # Directory for storing datasets (autopopulated after running nb)
    ├── models/              # Contains modules for SOC and SOH estimation
    ├── utils/               # Utility functions for data generation and processing
    ├── notebooks/           # Jupyter notebooks for interactive exploration
    │   └── soc_est_comparisons.ipynb  # Main notebook for testing and visualization
    ├── README.md            # Project overview and documentation
    └── requirements.txt     # List of dependencies required to run the project
```
## Getting Started

### Prerequisites

To run this project, you will need Python installed on your machine along with the required libraries. Current version utilizations neing utilzied locally are as follows:
- Python (3.11.7)
- Virtual environment created using Python 3.11 (Not required but suggested)

Otherwise, you can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Authors

- [@javaidb](https://www.github.com/javaidb)


## Acknowledgements

 - [Mendeley EIS Dataset 1](https://data.mendeley.com/datasets/n78tkm784n/1)
 - [Impedance.py + Sample Dataset](https://impedancepy.readthedocs.io/en/latest/getting-started.html)

Models built on TensorFlow and modeled on datasets [LG 18650HG2](https://data.mendeley.com/datasets/cp3473x7xv/3) and  [Panasonic 18650PF](https://data.mendeley.com/datasets/wykht8y7tg/1).

<p align="center">
  <img src="https://github.com/user-attachments/assets/832b4082-f77e-449e-8dfc-c60f540236a7" width="200" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/6cb16e14-da65-4f08-944d-1b967c563e4e" width="120" />
</p>

## Badges

[![IBM License](https://img.shields.io/badge/Certificate_ML-IBM-blue.svg)](https://www.credly.com/badges/6d82b78c-cade-4a4c-94cb-b7f89e142350/public_url)
[![SOC Estimation](https://img.shields.io/badge/Certificate_SOC-CU-c0ae88.svg)](https://coursera.org/share/b6b06ac95cd73bc569d8a6530130b154)
[![SOH Estimation](https://img.shields.io/badge/Certificate_SOH-CU-c0ae88.svg)](https://coursera.org/share/784a52ce9a135d3068b94ad406ab038a)
