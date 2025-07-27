from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Create dataset
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title("Quantum Dataset: Moons 🌓")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Classical NN
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

clf = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Classical NN Accuracy: {accuracy * 100:.2f}%")


# Plot decision boundary
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_decision_boundary(clf, X_scaled, y, "Classical NN Decision Boundary")



# Quantum Neural Network (QNN)
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

# Define circuit with separate input and weight parameters
num_inputs = 2
input_params = ParameterVector('x', length=num_inputs)
weight_params = ParameterVector('θ', length=num_inputs)

qc = QuantumCircuit(num_inputs)
for i in range(num_inputs):
    qc.h(i)
    qc.ry(input_params[i], i)
qc.cz(0, 1)
for i in range(num_inputs):
    qc.rz(weight_params[i], i)

# Observable (Z on each qubit)
observable = SparsePauliOp(["ZZ"])

# Build QNN
qnn = SamplerQNN(
    circuit=qc,
    input_params=input_params,
    weight_params=weight_params,
    observable=observable,
)

# QNN Classifier
qnn_clf = NeuralNetworkClassifier(
    neural_network=qnn,
    optimizer='COBYLA',
    maxiter=100
)

# Split data again (or reuse scaled split)
X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train QNN
qnn_clf.fit(X_train_q, y_train_q)

# Evaluate
y_pred_q = qnn_clf.predict(X_test_q)
accuracy_q = accuracy_score(y_test_q, y_pred_q)
print(f"💡 Quantum Neural Network Accuracy: {accuracy_q * 100:.2f}%")


# 🔥 Final Comparison Report
print("\n📊 COMPARISON REPORT:")
print("-----------------------------")
print(f"🔥 Classical NN Accuracy: {accuracy * 100:.2f}%")
print(f"🔮 Quantum NN Accuracy:   {accuracy_q * 100:.2f}%")

if accuracy_q > accuracy:
    print("✅ Quantum Model performed better!")
elif accuracy_q < accuracy:
    print("✅ Classical Model performed better!")
else:
    print("⚖️  Both models performed equally well!")
