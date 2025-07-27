# 🔮 Quantum vs Classical Machine Learning

This project compares a **classical neural network** with a **quantum neural network** using a simple 2D classification task (`make_moons` dataset).

> Made for CQHack25 using Qiskit 🧠⚛️  
> By [Sayed Jakaria Ahmed (Jack)](https://github.com/YOUR_USERNAME)

---

## 🎯 Objective

Compare the performance of a:
- 🤖 Classical Neural Network (Scikit-learn MLP)
- ⚛️ Quantum Neural Network (Qiskit's SamplerQNN)

on the same dataset to evaluate whether quantum models can compete with classical ones in basic ML tasks.

---

## 🧪 Technologies Used

- Python 🐍
- Qiskit + Qiskit Machine Learning
- Scikit-learn
- NumPy
- Matplotlib

---

## 📊 Results

| Model Type | Accuracy |
|------------|----------|
| Classical NN | ~97% ✅ |
| Quantum NN | ~91% 🧠 |

> ⚠️ Quantum NN slightly underperformed due to limited expressibility and noisy simulation.

---

## 📈 Dataset

We used `make_moons()` from scikit-learn with some noise added:
```python
X, y = make_moons(n_samples=200, noise=0.2)
