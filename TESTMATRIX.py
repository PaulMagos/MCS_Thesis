# %%
import numpy as np
import matplotlib.pyplot as plt
from tsl.datasets import AirQuality

synth_data = np.loadtxt('Datasets/PredictionDatasets/ASGGTM_AirQuality_ADJ.csv')
synth_data = np.where(synth_data<0.1, 0., synth_data) # Replace values less than or equal to threshold with zeroes (optional: replace this part as per requirement )
synth_data = synth_data + synth_data.T
for i in range(synth_data.shape[0]):
    synth_data[i, i] = 0 # setting diagonal elements to zero (optional: replace this part as per requirement)
    # Replace values less than or equal to threshold with zeroes (optional: replace this part as per requirement 
# %%
plt.imshow(synth_data, cmap='magma', interpolation = 'nearest'); 
plt.colorbar() ; 
plt.title('Adjusted Rand Index (ASGGTM Synthetic Data Set Adj.)'), 
plt.xlabel("Predicted"),
plt.ylabel("Actual")
# %%

data = AirQuality(impute_nans=True, small=True).get_connectivity(
    method='distance',
    threshold=0.9,
    layout='dense',
    include_self=False
)
data.shape

# %%
plt.imshow(data, cmap='magma', interpolation = 'nearest'); 
plt.colorbar() ; 
plt.title('Adjusted Rand Index (ASGGTM Synthetic Data Set Adj.)'), 
plt.xlabel("Predicted"),
plt.ylabel("Actual")