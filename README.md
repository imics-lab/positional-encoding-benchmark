

<!-- TITLE -->
# Positional Encoding Benchmark for Time Series Classification

This repository provides a comprehensive benchmark for evaluating different positional encoding techniques in Transformer models, specifically for time series classification tasks. The project includes implementations of several positional encoding methods and Transformer architectures to test their effectiveness on various time series datasets.



<!-- DESCRIPTION -->
## Description

  
This project aims to analyze how positional encodings impact Transformer-based models in time series classification. The benchmark includes both fixed and learnable encoding techniques and explores advanced approaches like relative positional encoding. The project evaluates performance on a diverse set of datasets from different domains, such as human activity recognition, financial data, EEG recordings, etc.


## Positional Encoding Methods:
  - Absolute Positional Encoding (APE)
  - Learnable Positional Encoding (LPE)
  - Relative Positional Encoding (RPE)
  - Temporal Pseudo-Gaussian augmented Self-Attention (TPS)
  - Temporal Uncertainty Positional Encoding (TUPE)
  - time Absolute Positional Encoding (tAPE)
  - efficient Relative Position Encoding (eRPE)
  - Stochastic Positional Encoding (SPE)


## Dependencies
- Python 3.10
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Pandas
- Scikit-learn
- tsai
- PyTorch


## Installation

To install and run the Positional Encoding Benchmark, follow these steps:

```bash
git clone https://github.com/imics-lab/positional-encoding-benchmark.git
cd positional-encoding-benchmark
pip install -r requirements.txt


<!-- CONTRIBUTING -->
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


<!-- CITATION -->
## Citation

```bibtex
TBD
```
