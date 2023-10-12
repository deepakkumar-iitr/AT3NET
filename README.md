## All Signals Point to Personality: A Dual-Pipeline LSTM-Attention and Symbolic Dynamics Framework for Predicting Personality Traits from Bioelectrical Signals
###Co-authors: Deepak Kumar, Pradeep Singh, Balasubramanian Raman

## Contents:
•	Overview of Project 
•	Dataset Description
•	Dependencies 
•	Problem-Solving Approach 
•	Results

## Overview of the Project:
In this work, we present a novel model for personality trait prediction using a dual-pipeline architecture. The model architecture leverages Long Short-Term Memory (LSTM) networks with batch normalization for capturing sequential dependencies in data and incorporates temporal attention heads for feature extraction. By combining these parallel pipelines, our model effectively utilizes both LSTM and attention mechanisms to create a comprehensive representation of input data. The model aims to predict the OCEAN (openness, conscientiousness, extraversion, agreeableness, and neuroticism) traits using physiological signals, including EEG, ECG, and GSR.

## Dataset Description:
We have tested our approach on two datasets, AMIGOS and ASCERTAIN. AMIGOS dataset incorporates multi-modal recordings of participants as they viewed emotional movie segments. Data collection took place in two distinct experimental scenarios: 1)40 participants were exposed to 16 brief emotional video clips, and 2)Participants viewed four extended video segments, with some viewing sessions conducted individually and others in groups.
The ASCERTAIN dataset is a multimodal repository tailored for personality recognition, involving data from 58 participants. Each participant was subjected to 36 video clips, each evoking different emotions. The dataset offers diverse modalities, including ECG, GSR, EEG signals, and Facial Landmark Trajectories.


## Problem-Solving Approach 
Our Approach can be summarized with the following architecture diagram-
![image](https://user-images.githubusercontent.com/79198655/190871162-e118a57b-b55f-4527-954e-29039675ec69.png)

## Results
We evaluated our approach on Two Datasets (i.e., ASCERTAIN and AMIGOS) using Symbolic Dynamics as a preprocessing approach, and then we used the AT3Net model to evaluate. 
More details can be found in our manuscript. 
