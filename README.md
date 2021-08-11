# ISP Self-Operated BGP Anomaly Detection Based on Weakly Supervised Learning

# Background
The Border Gateway Protocol (BGP) is the most important and irreplaceable protocol in the network, but its lack of routing authentication and validation makes it easy to be used by attackers to achieve routing leaks, routing hijacking, prefix hijacking, etc. Therefore, some third-party anomaly detection systems have emerged. However, the different methods adopted by these systems cannot effectively detect all types of BGP anomalies. Therefore, we propose a generalized framework for BGP anomaly detection based on adaptive distillation learning. In our scheme, we propose a self attention-based Long Short-Term Memory (LSTM) model to self-adaptively mine the differences among BGP anomaly categories, including both feature and time dimensions. Besides, in order to improve the adaptability and reliability of our model based on the existing anomaly data sets, we propose the approach to learn the experience from the other anomaly detection systems by knowledge distillation. Finally, we implement our prototype and demonstrate performance through comprehensive experiments.

# Install
## environments --install:
conda env create -f environments.yaml
## Usage:
In BGP_Anomaly_detection packet:
Anomaly_Detector is main program, where you can change your hyper-parameters and excute the detection.

Feature_Extractor is feature extraction module, which can extract features from BGP historical UPDATE packets by pybgpstream. 
Also you can change the codes to extract real-time BGP UPDATE packets.

Data_loader is our preprocessing modlue, which can further transform our dates to timestamp dates and add some metrics features.
Self_Attention_LSTM.py is our detection model.

The events_data packet contains our event lists.
