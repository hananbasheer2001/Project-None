# !pip install fsspec
# !pip install s3fs
# !pip install boto

import pandas as pd
import boto
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA

# import the csv file directly from an s3 bucket
data = pd.read_csv('s3://articledatas3/CanonicalCorrelationAnalysisData.csv')

# Split the data in X and Y
X = data[['PsychTest1',	'PsychTest2', 'YrsEdu', 'IQ', 'HrsTrn', 'HrsWrk']]
Y = data[['ClientSat',	'SuperSat',	'ProjCompl']]

# Instantiate the Canonical Correlation Analysis with 2 components
my_cca = CCA(n_components=2)

# Fit the model
my_cca.fit(X, Y)

# Obtain the rotation matrices
xrot = my_cca.x_rotations_
yrot = my_cca.y_rotations_

# Put them together in a numpy matrix
xyrot = np.vstack((xrot,yrot))

nvariables = xyrot.shape[0]

plt.figure(figsize=(15, 15))
plt.xlim((-1,1))
plt.ylim((-1,1))

# Show the Canonical correlations per dimension to decide on the number of dimensions
plt.barplot(model$cor, xlab = "Dimension", ylab = "Canonical correlations", ylim = c(0,1))

# Plot an arrow and a text label for each variable
for var_i in range(nvariables):
    x = xyrot[var_i,0]
    y = xyrot[var_i,1]
    plt.text(x,y,data.columns[var_i], color='red' if var_i >= 6 else 'blue')

    angle = np.linspace(0, 2 * np.pi, 150) 
    radius = (x**2 + y**2)**0.5
    x = radius * np.cos(angle) 
    y = radius * np.sin(angle)
    plt.plot(x, y)
    print(x)

    # plt.arrow(0,0,x,y)

plt.show()