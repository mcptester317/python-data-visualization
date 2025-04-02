# Python Data Visualization

A collection of Python scripts for data visualization using matplotlib, seaborn, and other visualization libraries.

## Overview

This repository contains a variety of Python visualization scripts for:
- Statistical analysis visualizations
- Comparison plots for machine learning models
- Heatmaps for statistical significance analysis
- Network and flow diagrams
- Custom plot styling and formatting

## Scripts

### 1. Statistical Visualization (`plots.py`)

Individual plots for statistical analysis:
- Normal distributions
- Box plots
- Mean plots with error bars
- Percentile comparisons

```bash
python plots.py --plot distribution  # Generate only the distribution plot
python plots.py --plot all           # Generate all plots separately
```

### 2. Model Comparison (`plot_accuracy.py`)

Visualize performance comparisons between different ML models:
- Bar charts for model accuracy
- Color-coded performance metrics
- Customizable styling

```bash
python plot_accuracy.py
```

### 3. BBQ Split Analysis (`plot_bbq_accuracy.py`)

Multi-panel plot for analyzing model performance across different BBQ (Bias Benchmark for QA) splits:
- Side-by-side comparison of multiple models
- Consistent scaling and formatting
- Legend with multiple columns

```bash
python plot_bbq_accuracy.py
```

### 4. Statistical Significance Heatmaps (`p_value_heatmap.py`)

Generate heatmaps to visualize statistical significance between datasets:
- Triangular format to avoid redundancy
- Color-coded significance levels
- Independent t-test comparisons

```bash
python p_value_heatmap.py
```

### 5. Flow Diagrams (`research_diagram.py`)

Create flow diagrams for research proposals or process visualization:
- Directed graphs with custom node shapes
- Color-coded process stages
- Edge labels for relationship descriptions

```bash
python research_diagram.py
```

## Requirements

- Python 3.6+
- matplotlib
- seaborn
- numpy
- pandas
- scipy
- graphviz (requires system installation)

Install dependencies:

```bash
pip install matplotlib seaborn numpy pandas scipy
```

For graphviz:
```bash
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz

# Windows
# Download from the Graphviz website
pip install graphviz
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.