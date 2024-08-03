# GPerturb: Additive, multivariate, sparse distributional regression model for  perturbation effect estimation
This repository hosts the implementation of GPerturb [(link)](), a Bayesian model identify and estimate interpretable sparse gene-level perturbation effects. 

<p align="center"><img src="https://github.com/hwxing3259/GPerturb/blob/main/visualisation/figure1.png" alt="GPerturb" width="900px" /></p>

## Core API Interface
Here we use the SciPlex2 dataset from [Lotfollahi et al 2023](https://github.com/theislab/CPA) as an example
### Load relavent datasets
```
adata = sc.read('SciPlex2_new.h5ad')

torch.manual_seed(3141592)
# load data:
my_conditioner = pd.read_csv("SciPlex2_perturbation.csv", index_col=0)
my_conditioner = my_conditioner.drop('Vehicle', axis=1)  # TODO: or retaining it
cond_name = list(my_conditioner.columns)
my_conditioner = torch.tensor(my_conditioner.to_numpy() * 1.0, dtype=torch.float)
my_conditioner = torch.pow(my_conditioner, 0.2)  # a power transformation of dosages

my_observation = pd.read_csv("SciPlex2.csv", index_col=0)
print(my_observation.shape)
my_observation = torch.tensor(my_observation.to_numpy() * 1.0, dtype=torch.float)

gene_name = list(pd.read_csv('SciPlex2_gene_name.csv').to_numpy()[:, 0])

my_cell_info = pd.read_csv("SciPlex2_cell_info.csv", index_col=0)
my_cell_info.n_genes = my_cell_info.n_genes/my_cell_info.n_counts
my_cell_info.n_counts = np.log(my_cell_info.n_counts)
cell_info_names = list(my_cell_info.columns)
my_cell_info = torch.tensor(my_cell_info.to_numpy() * 1.0, dtype=torch.float)
```

### Define and train Gaussian-GPerturb
```
output_dim = my_observation.shape[1]
sample_size = my_observation.shape[0]
hidden_node = 700  
hidden_layer = 4
conditioner_dim = my_conditioner.shape[1]
cell_info_dim = my_cell_info.shape[1]

lr_parametric = 1e-3  
tau = torch.tensor(1.).to(device)

parametric_model = GPerturb_gaussian(conditioner_dim=conditioner_dim, output_dim=output_dim, base_dim=cell_info_dim,
                               data_size=sample_size, hidden_node=hidden_node, hidden_layer_1=hidden_layer,
                               hidden_layer_2=hidden_layer, tau=tau)
parametric_model.test_id = testing_idx = list(np.random.choice(a=range(my_observation.shape[0]), size=my_observation.shape[0] // 8, replace=False))
parametric_model = parametric_model.to(device)

#  train the model from scratch 
parametric_model.GPerturb_train(epoch=250, observation=my_observation, cell_info=my_cell_info, perturbation=my_conditioner, 
                                lr=lr_parametric, device=device)
```

### Get fitted values on test set
```
fitted_vals = Gaussian_estimates(model=parametric_model, obs=my_observation[parametric_model.test_id], 
                                 cond=my_conditioner[parametric_model.test_id], cell_info=my_cell_info[parametric_model.test_id])
```

## Reproducing numerical examples
Codes for reproducing the LUHMES example: [Link](https://github.com/hwxing3259/GPerturb/blob/main/numerical_examples/LUHMES_GPerturb.ipynb)

Codes for reproducing the TCells example: [Link](https://github.com/hwxing3259/GPerturb/blob/main/numerical_examples/TCells_GPerturb.ipynb)

Codes for reproducing the SciPlex2 example: [Link](https://github.com/hwxing3259/GPerturb/blob/main/numerical_examples/SciPlex2_GPerturb.ipynb)

Codes for reproducing the Replogle et al 2022 example: [Link](https://github.com/hwxing3259/GPerturb/blob/main/numerical_examples/Replogle_GPerturb.ipynb)

Pre-trained models and datasets can be downloaded from: [Link](https://drive.google.com/drive/folders/1OqzcBbEL3HHOjoSQTynwRHhx2w8WVIrU?usp=share_link)
