# Introduction
This is the Army Research Laboratory (ARL) EEGModels project: A Collection of Convolutional Neural Network (CNN) models for EEG signal processing and classification, written in Keras and Tensorflow. The aim of this project is to

- provide a set of well-validated CNN models for EEG signal processing and classification
- facilitate reproducible research and
- enable other researchers to use and compare these models as easy as possible on their data

# Requirements

- Python  3.7
- tensorflow-gpu == 2.1.0
- Keras variable 'image_data_format' = "channels_first" in keras.json configuration file
- Data shape = (trials, kernels, channels, samples), which for the input layer, will be (trials, 1, channels, samples). 

To run the EEG/MEG ERP classification sample script, you will also need

- mne >= 0.17.1
- PyRiemann >= 0.2.5
- scikit-learn >= 0.20.1
- matplotlib >= 2.2.3

# Models Implemented

- EEGNet [[1]](http://stacks.iop.org/1741-2552/15/i=5/a=056013). Both the original model and the revised model are implemented.
- EEGNet variant used for classification of Steady State Visual Evoked Potential (SSVEP) Signals [[2]](http://iopscience.iop.org/article/10.1088/1741-2552/aae5d8)
- DeepConvNet [[3]](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
- ShallowConvNet [[3]](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)


# Usage

To use this package, place the contents of this folder in your PYTHONPATH environment variable. Then, one can simply import any model and configure it as


```python

from EEGModels import EEGNet, ShallowConvNet, DeepConvNet

model  = EEGNet(nb_classes = ..., Chans = ..., Samples = ...)

model2 = ShallowConvNet(nb_classes = ..., Chans = ..., Samples = ...)

model3 = DeepConvNet(nb_classes = ..., Chans = ..., Samples = ...)

```

Compile the model with the associated loss function and optimizer (in our case, the categorical cross-entropy and Adam optimizer, respectively). Then fit the model and predict on new test data.

```python

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
fittedModel    = model.fit(...)
predicted      = model.predict(...)

```

# EEGNet Feature Explainability

To reproduce the EEGNet single-trial feature relevance results as we reported in [[1]](http://stacks.iop.org/1741-2552/15/i=5/a=056013), download and install DeepExplain located [[here]](https://github.com/marcoancona/DeepExplain), which implements a variety of relevance attribution methods (both gradient-based and perturbation-based). A sketch of how to use it is given below:

```python
from EEGModels import EEGNet
from tensorflow.keras.models import Model
from deepexplain.tensorflow import DeepExplain
from tensorflow.keras import backend as K

# configure, compile and fit the model
 
model          = EEGNet(nb_classes = ..., Chans = ..., Samples = ...)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
fittedModel    = model.fit(...)

# use DeepExplain to get individual trial feature relevances for some test data (X_test, Y_test). 
# Note that model.layers[-2] points to the dense layer prior to softmax activation. Also, we use
# the DeepLIFT method in the paper, although other options, including epsilon-LRP, are available.
# This works with all implemented models. 

# here, Y_test and X_test are the one-hot encodings of the class labels and
# the data, respectively. 

with DeepExplain(session = K.get_session()) as de:
	input_tensor   = model.layers[0].input
	fModel         = Model(inputs = input_tensor, outputs = model.layers[-2].output)    
	target_tensor  = fModel(input_tensor)    

	# can use epsilon-LRP as well if you like.
	attributions   = de.explain('deeplift', target_tensor * Y_test, input_tensor, X_test)
	# attributions = de.explain('elrp', target_tensor * Y_test, input_tensor, X_test)	
```
