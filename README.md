# 🔍 Real-Time Anomaly Detection Platform

A comprehensive, production-ready platform for detecting anomalies in time series data streams, monitoring model drift, and analyzing market regimes. Built with advanced ML techniques inspired by sophisticated trading systems.

## 🚀 Features

### Multi-Method Anomaly Detection
- **Statistical Methods**: Z-Score, Modified Z-Score (MAD), IQR, Grubbs Test
- **Machine Learning**: Isolation Forest, Local Outlier Factor, One-Class SVM, Elliptic Envelope
- **Deep Learning**: Autoencoder-based reconstruction error detection
- **Ensemble**: Weighted combination of all methods for robust detection

### Model Drift Detection
- **Kolmogorov-Smirnov Test**: Distribution comparison
- **Population Stability Index (PSI)**: Bucket-based drift measurement
- **Jensen-Shannon Divergence**: Information-theoretic distance
- **Mean Shift Detection**: Location parameter monitoring
- **Variance Ratio Test**: Scale parameter monitoring

### Market Regime Analysis
- **Gaussian Mixture Models**: Unsupervised regime detection
- **Regime Classification**: Bullish, Bearish, Consolidation, High Volatility
- **Transition Monitoring**: Track regime changes over time
- **Probability Distribution**: Confidence in current regime

### Real-Time Monitoring
- **Live Dashboard**: Streaming data visualization
- **Alert System**: Configurable thresholds and notifications
- **Performance Metrics**: Comprehensive statistics and KPIs
- **Export Capabilities**: CSV, JSON report generation

## 📁 Project Structure

```
anomaly_detection_platform/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── config/
│   └── settings.yaml     # Configuration settings
├── modules/
│   ├── __init__.py
│   ├── statistical.py    # Statistical anomaly detection
│   ├── ml_detection.py   # ML-based detection
│   ├── drift_detection.py # Model drift monitoring
│   ├── regime_detection.py # Market regime analysis
│   └── alerting.py       # Alert management
├── utils/
│   ├── __init__.py
│   ├── data_generator.py # Synthetic data generation
│   ├── visualization.py  # Plotly chart creators
│   └── preprocessing.py  # Data preprocessing
└── tests/
    └── test_detectors.py # Unit tests
```

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- pip or conda

### Quick Start

```bash
# Clone or download the project
cd anomaly_detection_platform

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Dependencies

Core requirements:
- streamlit >= 1.28.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0
- plotly >= 5.15.0
- scikit-learn >= 1.3.0

Optional (for enhanced features):
- torch >= 2.0.0 (deep learning models)
- xgboost >= 1.7.0 (gradient boosting)
- shap >= 0.42.0 (model explainability)

## 📊 Usage

### 1. Data Input

The platform supports multiple data sources:

**Synthetic Data (Demo)**
```python
# Generated automatically for testing
data = generate_synthetic_data(n_points=500, anomaly_rate=0.05)
```

**CSV Upload**
- Upload via the sidebar interface
- Supports timestamp + value columns
- Auto-detects column types

**API Integration**
```python
# Configure in settings.yaml
data_source:
  type: api
  endpoint: "https://your-api.com/data"
  auth_token: "your-token"
```

### 2. Detection Methods

#### Statistical Detection
```python
from modules.statistical import AdvancedStatisticalDetector

detector = StatisticalAnomalyDetector(
    window_size=100,
    z_threshold=3.0,
    iqr_multiplier=1.5
)

is_anomaly, score, details = detector.detect(value)
```

#### ML Detection
```python
from modules.ml_detection import MLAnomalyDetector

detector = MLAnomalyDetector(contamination=0.1)
detector.fit(training_data)
anomalies, scores = detector.detect(new_data)
```

#### Drift Detection
```python
from modules.drift_detection import ModelDriftDetector

detector = ModelDriftDetector(drift_threshold=0.1)
detector.set_reference(reference_data)
drift_detected, score, details = detector.detect_drift(current_data)
```

### 3. Configuration

Edit `config/settings.yaml`:

```yaml
detection:
  statistical:
    z_threshold: 3.0
    iqr_multiplier: 1.5
    window_size: 100
    
  ml:
    contamination: 0.1
    models:
      - isolation_forest
      - local_outlier_factor
      - one_class_svm
      
  drift:
    reference_window: 500
    detection_window: 50
    threshold: 0.1

alerting:
  enabled: true
  channels:
    - console
    - webhook
  thresholds:
    critical: 0.8
    warning: 0.5

visualization:
  theme: dark
  refresh_rate: 1000  # ms
```

## 📈 Dashboard Tabs

### 1. Anomaly Detection Tab
- Real-time time series with anomaly highlighting
- Multi-method comparison chart
- Detection statistics and metrics
- Detailed breakdowns per method

### 2. Model Drift Tab
- Distribution comparison (reference vs current)
- Drift score over time
- Statistical test results (KS, PSI, JS)
- Automated drift alerts

### 3. Regime Analysis Tab
- Price chart colored by detected regime
- Regime probability distribution
- Volatility and momentum indicators
- Transition history

### 4. Alerts Tab
- Real-time alert feed
- Severity filtering (Critical/Warning/Info)
- Alert history and statistics
- Configuration options

### 5. Reports Tab
- Executive summary
- Data quality metrics
- Detailed statistical analysis
- Export functionality (CSV, JSON)

## 🧪 API Reference

### StatisticalAnomalyDetector

```python
class StatisticalAnomalyDetector:
    def __init__(self, window_size=100, z_threshold=3.0, iqr_multiplier=1.5):
        """
        Initialize statistical detector.
        
        Args:
            window_size: Rolling window for statistics
            z_threshold: Z-score threshold for anomaly
            iqr_multiplier: IQR multiplier for bounds
        """
        
    def detect(self, value) -> Tuple[bool, float, dict]:
        """
        Detect if value is anomalous.
        
        Returns:
            is_anomaly: Boolean flag
            score: Anomaly score (0-1)
            details: Dict with method-specific results
        """
```

### MLAnomalyDetector

```python
class MLAnomalyDetector:
    def __init__(self, contamination=0.1):
        """
        Initialize ML-based detector with ensemble of models.
        
        Args:
            contamination: Expected proportion of anomalies
        """
        
    def fit(self, X) -> bool:
        """Fit all models on training data."""
        
    def detect(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in data.
        
        Returns:
            is_anomaly: Array of binary predictions
            scores: Array of anomaly scores
        """
```

### ModelDriftDetector

```python
class ModelDriftDetector:
    def __init__(self, reference_window=500, detection_window=50, drift_threshold=0.1):
        """
        Initialize drift detector.
        
        Args:
            reference_window: Size of reference distribution
            detection_window: Size of current window
            drift_threshold: Score threshold for drift
        """
        
    def set_reference(self, data):
        """Set reference distribution."""
        
    def detect_drift(self, current_data) -> Tuple[bool, float, dict]:
        """
        Detect if distribution has drifted.
        
        Returns:
            drift_detected: Boolean flag
            score: Drift score (0-1)
            details: Dict with test results
        """
```

## 🔧 Extending the Platform

### Adding Custom Detectors

```python
from modules.base import BaseDetector

class CustomDetector(BaseDetector):
    def __init__(self, **params):
        super().__init__()
        self.params = params
        
    def fit(self, X):
        # Training logic
        pass
        
    def detect(self, X):
        # Detection logic
        return is_anomaly, score, details
```

### Adding Data Sources

```python
from utils.data_sources import DataSource

class CustomDataSource(DataSource):
    def connect(self, **kwargs):
        # Connection logic
        pass
        
    def fetch(self, start, end):
        # Fetch data for time range
        return pd.DataFrame(...)
```

## 📝 Best Practices

### 1. Threshold Tuning
- Start with default thresholds
- Monitor false positive rate
- Adjust based on domain knowledge
- Use cross-validation on historical data

### 2. Reference Data Selection
- Use representative normal data
- Avoid periods with known anomalies
- Update periodically (not too frequently)
- Consider seasonal patterns

### 3. Alert Management
- Set up escalation rules
- Avoid alert fatigue
- Document response procedures
- Track alert accuracy

### 4. Performance Monitoring
- Log detection latency
- Monitor memory usage
- Track model degradation
- Schedule regular retraining

## 🐛 Troubleshooting

### Common Issues

**High False Positive Rate**
- Increase z_threshold
- Adjust contamination parameter
- Check for data quality issues
- Consider longer reference window

**Missing Anomalies**
- Decrease detection thresholds
- Add more detection methods
- Verify data preprocessing
- Check for concept drift

**Slow Performance**
- Reduce window sizes
- Disable unnecessary methods
- Use sampling for large datasets
- Enable caching

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## 📞 Support

For issues and feature requests, please use the GitHub Issues tab.

---

Built with ❤️ using Streamlit, Plotly, and scikit-learn
