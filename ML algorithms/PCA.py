import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import seaborn as sns
# %% Load
dfForTraining = pd.read_csv('../Dataframes/test.csv').iloc[:, 1:]
#%% Delete diagnostic column
y = dfForTraining.loc[:, ['diagnostic']]
dfForTraining.pop('diagnostic')
dfForTraining = dfForTraining.iloc[:, :]
#%% Binarize diagnostic column
lb = LabelBinarizer()
y = lb.fit_transform(y)
#%%
dfForTraining.pop('marker')
scaler = StandardScaler()
scaler.fit(dfForTraining.iloc[:, :])
scaled_data = scaler.transform(dfForTraining.iloc[:, :])
#%% PCA
n_components = 9
pca = PCA(n_components=n_components)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
columns = []
for i in range(0,n_components):
    columns.append(f'Component{i+1}')
columns.append('diagnostic')
data = np.concatenate((x_pca, y), axis=1)
componentsDf = pd.DataFrame(data, columns=columns)
print(pca.explained_variance_ratio_.sum())
componentsDf.to_csv('../Dataframes/PCA_Reduced_Features.csv')
#%%
sns.pairplot(componentsDf);
# %%
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0], x_pca[:,1], c=y.ravel(), cmap='plasma')
plt.xlabel('First component')
plt.ylabel('Second component')