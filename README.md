<h1> Generalization in Diversity-Aware Argument Strength Models </h1>
<h5> Master Thesis <br>
Siddharth Bhargava </h5>

---

<h2> Brief Description </h2>

Through this study, I seek to explore and understand how argument strength
estimation can be generalized using the work and data from a diverse range
of sources. I define various experiments and scenarios that study the diversity
aspect of the different domains and underlines the connection between gen-
eralization and diversity, focusing the employment of multi-task learning as a
method to share information between the different interpretations of argument
strength.

Read Thesis: https://www.overleaf.com/read/sppbzngpxzzn

---
<h2> Setup and Environment </h2>

Software and Tools:- 
- OS: Ubuntu 20.04.4 LTS
- GPUs: NVIDIA GeForce RTX 2080 Ti (4 units)
- Python version: 3.8.5
- MLflow Client: http://mlflow.dbs.ifi.lmu.de:5000/
- PyCharm 2020 was used for the purpose of setting the environment, writing the scripts, debugging the code and deploying it onto the server.

All the required libraries and packages for setting up the environment can be installed using the following code:

`pip install -r requirements.txt`

---

<h2> Repository Setup </h2>

- <b>Hyper-parameter-optimization</b> directory contains executable code using Pytorch and ray tune. The Readme file in the directory holds instructions for execution of the code.

- <b> pytorch-lightning </b> directory contains executable code using Pytorch Lightning setup. The Readme file in the directory holds instructions for execution of this code.

- <b> raw-data </b> folder contains the data sets as downloaded from the respective sources. The initial data preparation and exploratory data analysis is presented in the jupyter notebook provided in the folder.
