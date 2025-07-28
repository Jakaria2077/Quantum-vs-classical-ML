# 🔮 Quantum vs Classical Machine Learning

A fun comparison between a **Classical Neural Network** and a **Quantum Neural Network** using a simple 2D classification task — the classic `make_moons` dataset 🌙✨

> Made for **CQHack25** using **Qiskit** 🧠⚛️  
> By **Sayed Jakaria Ahmed (Jack)**

---

## 🎯 Objective

To compare the performance of:

- 🤖 **Classical Neural Network** (Scikit-learn `MLPClassifier`)  
- ⚛️ **Quantum Neural Network** (Qiskit's `SamplerQNN`)

...on the same dataset and see whether quantum models can keep up with classical ones in basic ML tasks.

---

## 🧪 Technologies Used

- Python 🐍  
- Qiskit + Qiskit Machine Learning ⚛️  
- Scikit-learn 🤖  
- NumPy 🔢  
- Matplotlib 📊  
- GitHub + VSC 💻

---

## 📊 Results

| Model Type         | Accuracy |
|--------------------|----------|
| Classical NN 🤖     | ~97% ✅    |
| Quantum NN ⚛️       | ~91% 🧠    |

> ⚠️ Quantum NN slightly underperformed due to limited expressibility and noisy simulation.

---

## ▶️ How to Run

**1. Clone the repo:**
```bash
git clone https://github.com/your-username/QuantumEdge.git
cd QuantumEdge
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the code:**
```bash
python quantum_edge.py
```

**You'll see:**
- 📈 Plot 1: Classical Neural Network decision boundary  
- 📉 Plot 2: Quantum Neural Network decision boundary  
- ✅ Accuracy and comparison printed in terminal

---

## 🌟 Inspiration

I’ve always been fascinated by the strange, mysterious quantum world — how particles can be in superposition or entangled across time and space.  
That curiosity led me to discover quantum computing.

This project is my **first step** into merging quantum mechanics and AI — and I’ve loved every second of it! ❤️⚛️

---

## 🔧 Challenges Faced

- Learning Qiskit from scratch in a few days  
- Understanding how Quantum Neural Networks actually work  
- Fixing broken imports and deprecated functions  
- Making a fair comparison between two totally different architectures

---

## 🚀 Future Work

- Run this on real quantum hardware (IBM Qiskit backends)  
- Explore **Variational Quantum Classifiers (VQC)**  
- Boost Quantum NN accuracy with a better ansatz  
- Build a full UI or deploy a web app to visualize live comparison
