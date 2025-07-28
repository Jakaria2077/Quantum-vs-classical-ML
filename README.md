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

▶️ How to Run
Clone the repo:

bash
Copy
Edit
git clone https://github.com/your-username/QuantumEdge.git
cd QuantumEdge
Install requirements:

bash
Copy
Edit
pip install -r requirements.txt
Run the main script:

bash
Copy
Edit
python quantum_edge.py
What happens:

Two plots will be shown:

Decision boundary of Classical Neural Net

Decision boundary of Quantum Neural Net

You'll see comparison printed in the terminal.

🌟 Inspiration
I’ve always been fascinated by the strange, mysterious quantum world — how particles can be in superposition or entangled across time and space. That fascination led me to discover quantum computers.
This project is my first attempt at combining quantum mechanics and machine learning — and I loved every second of it ⚛️❤️

🔧 Challenges Faced
Learning Qiskit from scratch in a short time

Understanding how Quantum Neural Networks work

Fixing broken imports and old deprecated functions

Aligning outputs of classical and quantum models fairly

🚀 Future Work
Test this on real quantum hardware using IBM Qiskit backend

Explore Variational Quantum Classifiers (VQC)

Improve quantum model accuracy with better ansatz

Build a proper UI or web dashboard for visualization

🔗 Try it Out
🔗 GitHub Repo
