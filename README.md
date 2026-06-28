# Jet_tagging

<div style="text-align: justify;">
The paper has been published here: http://arxiv.org/abs/2606.24637. Jet charge computation using classical and quantum machine learning is an innovative approach in high-energy particle physics, specifically in the analysis of data from particle colliders like the Large Hadron Collider (LHC). The jet charge is a weighted sum of the electric charges of the particles within a jet, typically weighted by their transverse momenta (example plot below). This quantity provides important insights into the properties of the originating quark or gluon and helps in distinguishing between different types of jets, such as those originating from up-type versus down-type quarks. By leveraging machine learning algorithms, we can enhance the precision and efficiency of jet charge determination. These algorithms can learn complex patterns and correlations within the high-dimensional data, outperforming traditional methods. Techniques such as deep neural networks, gradient boosting, and other supervised learning approaches are trained on simulated datasets to accurately predict the jet charge. This advancement not only improves the accuracy of jet classification but also aids in probing the fundamental interactions and properties of particles, potentially uncovering new physics beyond the Standard Model. Dataser for this analysis is available in https://doi.org/10.5281/zenodo.15790634.
</div>

<p align="center">
<img width="900" alt="Screen Shot 2024-08-06 at 4 38 10 PM" src="https://github.com/user-attachments/assets/219927a9-389a-47a4-bf94-7a86bab0a64a">
</p>

## Running the Code to Generate Ntuples and Apply Machine Learning Models
This guide outlines the sequence of steps to generate ntuples using MadGraph and PYTHIA and to run quantum and classical machine learning models. Follow the instructions below to set up and execute the process.

### Prerequisites
Ensure you have the following tools installed:
- MadGraph
- PYTHIA
- Python with necessary libraries (e.g., NumPy, SciPy, TensorFlow/PyTorch, Scikit-learn, Qiskit)

### Step-by-Step Instructions
1. Generate Processes with MadGraph:
Inside the`uubar` notebook, use the `write_mg_cards` function to generate events for the following processes in MadGraph:
- `p p > u g`
- `p p > u~ g`
The function allows for the generation of an arbitrary number of events.
```
write_mg_cards('pp_to_ug', num_events=10000)
write_mg_cards('pp_to_u~g', num_events=10000)
```

2. Run PYTHIA and Produce Ntuples:
Utilize the `run_pythia_get_images` function to simulate the events with PYTHIA and produce the ntuples. This function extracts essential variables such as the jet's 4-vector, charge weighted with transverse momentum and mass, etc.
```
run_pythia_get_images('pp_to_ug', output_file='ntuples_ug.root')
run_pythia_get_images('pp_to_u~g', output_file='ntuples_ug_bar.root')
```

3. Apply Classical Machine Learning Models:
Several classical machine learning models are applied to classify up and anti-up jets using the input variables:
- Deep Neural Network (DNN)
- Support Vector Machine (SVM)
Hyperparameters for these models are optimized during training.
```
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential

# Example: Training an SVM with hyperparameter optimization
svm = SVC(kernel='linear')
parameters = {'C': [1, 10, 100]}
clf = GridSearchCV(svm, parameters)
clf.fit(X_train, y_train)

# Example: Training a DNN
model = Sequential([...])  # Define your DNN architecture here
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
```

4. Prepare Data for CNN:

Use the `make_image_leading_jet` and `make_image_event` functions to extract information from the jet's tracks. The output numpy histogram2d arrays are fed into a Convolutional Neural Network (CNN) model for training.
```
leading_jet_image = make_image_leading_jet(jet_tracks)
event_image = make_image_event(event_tracks)

# Example: Training a CNN
cnn_model = Sequential([...])  # Define your CNN architecture here
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(image_data, labels, epochs=50, batch_size=32, validation_split=0.2)
```

5. Run Graph Neural Networks (GNN):
To run the Graph Neural Network (GNN) and extract the best nodes, execute the [server_GNN](./server_GNN.py) script.

6. Quantum Machine Learning Models:
- Support Vector Machine with Quantum Kernel:
  The SVM with a quantum kernel is implemented in the [QSVM](./server_QSVM.py) file.
  ```
  from qiskit_machine_learning.algorithms import QSVM
  
  qsvm = QSVM(feature_map, training_input, test_input, datapoints)
  qsvm.run(quantum_instance)
  ```
- Variational Quantum Classifier (VQC):
  The VQC is also included in the [VQC](./server_VQC.py) file.
  ```
  from qiskit_machine_learning.algorithms import VQC
  
  vqc = VQC(quantum_instance, var_form, optimizer, feature_map)
  vqc.fit(X_train, y_train)
  ```
- Quantum Neural Network (QNN):
  The QNN implementation can be found in the [QNN](./server_QNN.py) file as well.
  ```
  from qiskit_machine_learning.algorithms import QNN

  qnn = QNN(quantum_instance, var_form, feature_map)
  qnn.fit(X_train, y_train)
  ```
  
