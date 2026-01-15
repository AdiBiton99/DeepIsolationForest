# Deep Isolation Forest (DIF) – Reproducible Setup (Python 3.11)

This repository provides a working and reproducible setup of **Deep Isolation Forest (DIF)** for anomaly detection,  
based on the original implementation by Xu et al. (TKDE 2023).

The environment was adapted to run on **Python 3.11** and tested on Windows with CPU execution.

---

## 📄 Original Paper

**Deep Isolation Forest for Anomaly Detection**  
Hongzuo Xu, Guansong Pang, Yijie Wang, Yongjun Wang  
IEEE Transactions on Knowledge and Data Engineering (TKDE), 2023

- Paper: https://arxiv.org/abs/2206.06602  
- IEEE: https://ieeexplore.ieee.org/document/10108034  
- Original repository: https://github.com/xuhongzuo/deep-iforest

---

## ⚙️ Environment

- Python: 3.11  
- OS: Windows / Linux  
- Device: CPU (default)

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create virtual environment (Python 3.11)
```bash
py -3.11 -m venv .venv
```

### 3. Activate the environment

Windows:
```bash
.venv\Scripts\activate
```

Linux / macOS:
```bash
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the project

Example:
```bash
python main.py --dataset shuttle_16 --device cpu
```

Another example:
```bash
python main.py --dataset pageblocks_16 --device cpu
```

---

## 🧠 Using DIF as a library

```python
from algorithms.dif import DIF

model_configs = {
    'n_ensemble': 50,
    'n_estimators': 6
}

model = DIF(**model_configs)
model.fit(X_train)
scores = model.decision_function(X_test)
```

---

## 📁 Project structure (simplified)

```
.
├── algorithms/
│   └── dif.py
├── data/
│   └── tabular/
├── main.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## ⚠️ Notes

- This version was adapted to support **Python 3.11**.
- PyTorch and PyTorch-Geometric dependencies are installed via precompiled wheels to ensure compatibility on Windows.
- The project runs on CPU by default.

---

## 📚 Citation

```bibtex
@ARTICLE{xu2023deep,
  author={Xu, Hongzuo and Pang, Guansong and Wang, Yijie and Wang, Yongjun},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  title={Deep Isolation Forest for Anomaly Detection},
  year={2023},
  pages={1-14},
  doi={10.1109/TKDE.2023.3270293}
}
```

---

## 🙏 Acknowledgments

This project is based on the original implementation by:

Xu, Pang, Wang & Wang – Deep Isolation Forest (TKDE 2023)

Original repository:  
https://github.com/xuhongzuo/deep-iforest
