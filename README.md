

<!-- TITLE -->
# Positional Encoding Benchmark for Time Series Classification

This repository provides a comprehensive evaluation framework for positional encoding methods in transformer-based time series models, along with implementations and benchmarking results.

Our work is available on arXiv: [Positional Encoding in Transformer-Based Time Series Models: A Survey](https://arxiv.org/abs/2502.12370)

## Models

We present a systematic analysis of positional encoding methods evaluated on two transformer architectures:
1. [Multivariate Time Series Transformer Framework (TST)](https://github.com/gzerveas/mvts_transformer)
2. Time Series Transformer with Patch Embedding 



### Positional Encoding Methods
We implement and evaluate eight positional encoding methods:

| Method | Type | Injection Technique | Parameters |
|--------|------|-------------------|------------|
| Sinusoidal PE | Absolute | Additive | 0 |
| Learnable PE | Absolute | Additive | L×d |
| RPE | Relative | MAM | 2(2L-1)dl |
| tAPE | Absolute | Additive | Ld |
| eRPE | Relative | MAM | (L²+L)l |
| TUPE | Rel+Abs | MAM | 2dl |
| ConvSPE | Relative | MAM | 3Kdh+dl |
| T-PE | Rel+Abs | Combined | 2d²l/h+(2L+2l)d |

Where:
- L: sequence length
- d: embedding dimension
- h: number of attention heads
- K: kernel size
- l: number of layers

## Dataset Characteristics

| Dataset | Train Size | Test Size | Length | Classes | Channels | Type |
|---------|------------|-----------|---------|----------|-----------|------|
| Sleep | 478,785 | 90,315 | 178 | 5 | 1 | EEG |
| ElectricDevices | 8,926 | 7,711 | 96 | 7 | 1 | Device |
| FaceDetection | 5,890 | 3,524 | 62 | 2 | 144 | EEG |
| MelbournePedestrian | 1,194 | 2,439 | 24 | 10 | 1 | Traffic |
| SharePriceIncrease | 965 | 965 | 60 | 2 | 1 | Financial |
| LSST | 2,459 | 2,466 | 36 | 14 | 6 | Other |
| RacketSports | 151 | 152 | 30 | 4 | 6 | HAR |
| SelfRegulationSCP1 | 268 | 293 | 896 | 2 | 6 | EEG |
| UniMiB-SHAR | 4,601 | 1,524 | 151 | 9 | 3 | HAR |
| RoomOccupancy | 8,103 | 2,026 | 30 | 4 | 18 | Sensor |
| EMGGestures | 1,800 | 450 | 30 | 8 | 9 | EMG |

## Dependencies
- Python 3.10
- PyTorch 2.4.1+cu121
- NumPy
- Scikit-learn
- CUDA 12.2 

## Clone and Installation

```bash
# Clone the repository
git clone https://github.com/imics-lab/positional-encoding-benchmark.git
cd positional-encoding-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run benchmark with default config
python examples/run_benchmark.py

# Or with custom config
python examples/run_benchmark.py --config path/to/custom_config.yaml
```

## Results

Our experimental evaluation encompasses eight distinct positional encoding methods tested across eleven diverse time series datasets using two transformer architectures.

### Key Findings

#### 1. Sequence Length Impact
- **Long sequences** (>100 steps): 5-6% improvement with advanced methods
- **Medium sequences** (50-100 steps): 3-4% improvement
- **Short sequences** (<50 steps): 2-3% improvement

#### 2. Architecture Performance
- **TST**: More distinct performance gaps
- **Patch Embedding**: More balanced performance among top methods

#### 3. Average Rankings
- **SPE**: 1.727 (batch norm), 2.090 (patch embed)
- **TUPE**: 1.909 (batch norm), 2.272 (patch embed)
- **T-PE**: 2.636 (batch norm), 2.363 (patch embed)

### Performance Analysis

#### Biomedical Signals (EEG, EMG)
- TUPE achieves highest average accuracy
- SPE shows strong performance
- Both methods demonstrate effectiveness in capturing long-range dependencies

#### Environmental and Sensor Data
- SPE exhibits superior performance
- TUPE maintains competitive accuracy
- Relative encoding methods show improved local pattern recognition

<!-- CONTRIBUTING -->
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


<!-- CITATION -->
## Citation

```bibtex
@article{irani2025positional,
  title={Positional Encoding in Transformer-Based Time Series Models: A Survey},
  author={Irani, Habib and Metsis, Vangelis},
  journal={arXiv preprint arXiv:2502.12370},
  year={2025}
}
```
