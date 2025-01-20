Frequency Encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Frequency encoding employs input features to determine the frequency of spikes. The frequency of spikes emitted by a neuron contains all the information. The measure of spike frequency can be completed simply by counting the number of spikes in a time interval, which is the temporal average. The frequency of neuronal spike emission can be understood as the ratio of the average number of spikes observed within a specific time interval T to the time T.

Temporal Encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Temporal encoding captures the precise spike timing information of neurons; a single spike carries more meaning than frequency codes that rely on emission frequency. Any spike sequence has a corresponding temporal firing pattern, hence the information related to stimuli might be expressed via the precise spike timing. Specifically, temporal encoding considers that the temporal structure of neuronal spike sequences carries stimulus signals on millisecond or even smaller scales, rather than merely the average firing frequency.

Spike delay coding is a type of temporal encoding, where the method encodes information within the precise spike timing structures of a set of interrelated spikes.

Population Encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Distinct from methods that encode through a single neuron, neural information can also be encoded through the activity of multiple neurons. Population encoding is a method of encoding stimulus signals using the collective response of a cluster of neurons. In population encoding, each neuron has a unique spike response distribution to a given input stimulus, with the combined response of neuron groups representing the overall information input.

In this case, population encoding is achieved using a Gaussian tuning curve to convert an analog quantity into a set of spike times for different neurons. A neuron covers a certain range of the analog quantity in the form of a Gaussian function, and the height of the Gaussian function corresponding to a certain value of the analog quantity determines the spike timing of the neuron.

Dataset and Network Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case, the MNIST dataset is encoded and converted into multi-shot spike data as model input. The model adopts

a sequential model of the DVS-MNIST dataset, containing ConvLif x3, FC x3.