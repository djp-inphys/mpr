
# MPR: Meta Pattern Recognition

**meta-pattern recognition** is a novel approach to enhance statistical decision-making and machine learning, aiming to address limitations of traditional pattern recognition. The key idea is to apply ML techniques recursively to the problem of pattern recognition itself, creating a hierarchical system where each level represents another application of the pattern recognition process.

### Key Features and Advantages

*   **Overcomes Limitations:** Meta-pattern recognition aims to overcome challenges like the difficulty in choosing appropriate classifiers and the peaking phenomenon (curse of dimensionality) that affects the performance of pattern recognition systems with high-dimensional data.
*   **Hierarchical Structure:** The hierarchical structure blurs the distinction between data representation (feature extraction/selection) and classification, allowing the system to automatically determine the most suitable features and classification schemes for a given dataset.
*   **Adaptability:** The system can expand or contract its internal complexity to match the complexity of the problem, adhering to Ashby's law of requisite variety.
*   **Automation:** It automates the preprocessing of input data, dimensionality reduction, and selection of optimal classifiers, minimising the need for human intervention.
*   **Multi-level Learning:** The system incorporates multiple levels of learning, enabling it to adapt and optimise its performance based on the data it processes.
*   **Wide Applicability:** The approach is applicable to a wide range of problems, as long as the data is represented as a set of real numbers and contains sufficient complexity.
*   **Improved Classification Efficiency:** Meta-pattern recognition can significantly improve the accuracy of classification, particularly in challenging scenarios like stock market prediction and medical diagnosis.
*   **Integration of AI Approaches:** It bridges the gap between rule-based (expert systems) and connectionist (neural networks) AI systems, offering a unified framework for decision-making.

### Core Concepts

*   **Optimal Subset:** The method identifies the most informative subset of features that best represent the underlying patterns in the data, reducing redundancy and noise.
*   **Classification Space:** The output of multiple classifiers is treated as a new dataset, and an optimal subset of classifiers is selected to create a classification space, capturing the relationships between different classification approaches.
*   **Enhanced Classification Space:** The classification space is further enhanced by incorporating the true class labels, enabling the system to learn the relationships between classifiers and the actual classes.
*   **Orthogonal Truth Space:** An orthogonal decision space is constructed around the true class labels, minimising the number of classifiers needed for accurate classification.

### Potential Applications

The core ideas explored in MPR, such as dimensionality reduction, feature extraction, and the quest for optimal representations of data, are fundamental to the success of contemporary AI algorithms. The MPR's emphasis on extracting meaningful information from complex data and making informed decisions based on patterns aligns well with the goals of modern AI, which strives to develop intelligent systems capable of understanding and interpreting the world around them.

The MPR's exploration of the relationship between data representation and classification, as well as its focus on minimising noise and redundancy in data, resonates with current efforts in AI to develop robust and efficient models that can generalise well to new, unseen data. The concept of "meta-pattern recognition," which involves applying pattern recognition techniques recursively to optimise the pattern recognition process itself, foreshadows the current trend in AI towards developing more adaptable and self-learning systems.

The pursuit of optimal data representations, efficient feature extraction, and effective classification strategies continues to drive advancements in AI research and applications. 

**In summary**, meta-pattern recognition presents a promising direction for advancing the field of pattern recognition by automating and optimising the process of feature selection, classifier selection, and dimensionality reduction, leading to more accurate and efficient decision-making systems. 

### Sources and Related Content

*   The concepts of dimensionality reduction and feature extraction are central to modern machine learning techniques such as Principal Component Analysis (PCA), t-SNE, and autoencoders.
*   The MPR's focus on minimising noise and redundancy in data is echoed in current research on data cleaning, pre-processing, and regularisation techniques.
*   The idea of "meta-pattern recognition" and its hierarchical approach to pattern recognition can be seen as a precursor to modern meta-learning and AutoML techniques, which aim to automate the process of selecting and optimising machine learning models.
*   The MPR's application of pattern recognition to medical imaging is still an active area of research, with deep learning models now playing a major role in image analysis and diagnosis.
---

## Table of Contents

- [MPR: Meta Pattern Recognition](#mpr-meta-pattern-recognition)
    - [Key Features and Advantages](#key-features-and-advantages)
    - [Core Concepts](#core-concepts)
    - [Potential Applications](#potential-applications)
    - [Sources and Related Content](#sources-and-related-content)
  - [Table of Contents](#table-of-contents)
  - [Key Concepts of MPR](#key-concepts-of-mpr)
  - [Features](#features)
  - [Dependencies](#dependencies)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
  - [Configuration](#configuration)
  - [Contributing](#contributing)

## Key Concepts of MPR

* **Optimal Subset Selection:** The pipeline aims to identify and utilise the most informative subset of features, addressing the "curse of dimensionality" and improving classification performance.
* **Classification Space:** The outputs of multiple classifiers are combined into a "classification space," enabling a more robust and adaptable classification strategy.
* **Hierarchical Learning:** The framework has the potential to be extended to support hierarchical learning, where the system recursively refines its feature selection and classification processes.

## Features

- **Classifier Support**: The pipeline supports various classifiers from scikit-learn.
- **Leave-One-Out Cross-Validation**: Employs LOO for robust model evaluation.
- **Hyperparameter Tuning**: Utilises grid search for hyperparameter optimisation.
- **Dimensionality Reduction**: PCA is used for feature extraction and reduction.
- **Parallel Processing**: Leverages multithreading for improved efficiency.
- **KY Data Analysis**: Performs PCA and computes Kittler-Young (KY) scores, offering 3D visualisations for analysis (potentially related to the MPR concept of "orthogonal truth space").

## Dependencies

- `numpy`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `pickle`

Install via `pip`:

```bash
pip install numpy scikit-learn pandas matplotlib
```

## Project Structure

```
├── config_submit.py        # Configuration file
├── main.py                 # Main execution script
├── process.py              # Core training and evaluation logic
├── representation.pkl      # Serialized PCA stages (output)
├── models.pkl              # Serialized trained models (output)
└── README.md               # Project documentation
```

## Usage

1. **Configure**: Update `config_submit.py` with your settings.
2. **Train**: Run `python main.py` with `run_type` set to `"build"`.
3. **Predict**: (Implementation pending) Set `run_type` to `"run"` and execute `python main.py`.

## Configuration

Key settings in `config_submit.py`:

- `datapath`: Path to your data.
- `n_stages`: Number of hierarchical classification stages.
- `n_dim`: Number of PCA dimensions to retain.
- `run_type`: `"build"` for training, `"run"` for prediction.
- `clsfrs` and `clsfrs_params`: Define the classifiers and their hyperparameter grids.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests.

**Note:**

* The current code focuses on binary classification.
* The prediction ("run") mode is not yet fully implemented.
* The KY data analysis and visualisation functions may be related to the MPR concept of "orthogonal truth space," warranting further exploration.

