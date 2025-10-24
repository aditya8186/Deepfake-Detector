# Deepfake Detection Project

This project focuses on detecting deepfake videos using deep learning techniques. The project is structured to handle data processing, model training, and deployment.

## Project Structure

```
deepfake_detection/
├── data/                    # Data files
│   ├── raw/                 # Raw dataset
│   │   ├── real/            # Real video samples
│   │   └── fake/            # Fake/Deepfake video samples
│   └── processed/           # Processed data (frames, features, etc.)
├── notebooks/               # Jupyter notebooks for EDA and testing
├── src/                     # Source code
│   ├── data/                # Data loading and preprocessing
│   ├── models/              # Model architectures
│   └── utils/               # Utility functions
├── tests/                   # Test cases
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd deepfake_detection
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Place real videos in `data/raw/real/`
   - Place fake/deepfake videos in `data/raw/fake/`

## Usage

1. Run Jupyter Notebook for EDA:
   ```bash
   jupyter notebook
   ```
   Then open the notebooks in the `notebooks/` directory.

## License

This project is licensed under the MIT License.
