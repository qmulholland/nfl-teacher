# Defensive Scout: AI Formation Classifier
An automated defensive scouting engine that classifies defensive formations from schematic dot diagrams. Built to emulate the NFL Next Gen Stats (NGS) framework for player identification and coverage responsibility.

# Project Overview
This project automates the identification of complex defensive fronts (e.g., Nickel 2-4, 3-4 Under, Dime 2-3 Odd) using a Convolution Neural Network (CNN). The model identifies defensive formation with high precision and sub-200ms latency through processing player coordinates.

## Framework
Built on the Roboflow Inference API with a custom Python-based Live Dashboard

# Technical Architecture
1. The Data Model
The dataset consists of 803 diagrams representing the different formations from the most recent Madden.
 - Input: 640x640 defensive formation diagrams
 - Classes: Extensive categories including Nickel,     Dime, 3-4, 4-3, and various "Mug" and "Load" fronts

2. Software Stack
 - Language: Python 3.12
 - AI Engine: Roboflow Inference SDK
