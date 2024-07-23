# GPerturb: Additive, multivariate, sparse distributional regression model for  perturbation effect estimation
This repository hosts the implementation of GPerturb [(link)](), a Bayesian model identify and estimate interpretable sparse gene-level perturbation effects. 

<p align="center"><img src="https://github.com/hwxing3259/GPerturb/blob/main/visualisation/figure1.png" alt="GPerturb" width="900px" /></p>

## Core API Interface
Here we use the Reoplogle dataset from [Bereket & Karaletsos, NeurIPS 2023](https://arxiv.org/abs/2311.02794) as an example
```
# ############################################# load dataset ##########################################
adata = sc.read_h5ad('replogle.h5ad')
adata.obs['n_feature'] = (adata.X > 0).sum(1)

my_observation = adata.X
gene_name = list(adata.var.gene_name)
my_observation = torch.tensor(my_observation * 1.0, dtype=torch.float)

my_cell_info = adata.obs[['core_adjusted_UMI_count', 'mitopercent', 'n_feature', 'core_scale_factor']]
my_cell_info = torch.tensor(my_cell_info.to_numpy() * 1.0, dtype=torch.float)
my_cell_info[:, 2] = my_cell_info[:, 2] / my_cell_info[:, 0]
my_cell_info[:, 0] = np.log(my_cell_info[:, 0])

pathways = adata.uns['pathways']

my_conditioner = pd.get_dummies(adata.obs['gene'])
my_conditioner = my_conditioner.drop('non-targeting', axis=1)
cond_name = list(my_conditioner.columns)
my_conditioner = torch.tensor(my_conditioner.to_numpy() * 1.0, dtype=torch.float)

# ##################### define and train ZIP=GPerturb #################################################
output_dim = my_observation.shape[1]
sample_size = my_observation.shape[0]
hidden_node = 1000
hidden_layer = 4
conditioner_dim = my_conditioner.shape[1]
cell_info_dim = my_cell_info.shape[1]
lr_parametric = 5e-4  
tau = torch.tensor(1.).to(device)

parametric_model = GPerturb_ZIP(conditioner_dim=conditioner_dim, output_dim=output_dim,
                                base_dim=cell_info_dim, data_size=sample_size,
                                hidden_node=hidden_node, hidden_layer_1=hidden_layer,
                                hidden_layer_2=hidden_layer, tau=tau)
parametric_model = parametric_model.to(device)

# train the model from scratch
parametric_model.GPerturb_train(epoch=300, observation=my_observation, cell_info=my_cell_info, perturbation=my_conditioner, 
                                lr=lr_parametric, device=device)
```

## Reproducing numerical examples
Codes for reproducing the LUHMES example: [Link](https://github.com/hwxing3259/GPerturb/blob/main/demo/LUHMES_GPerturb.ipynb)

Codes for reproducing the TCells example: [Link](https://github.com/hwxing3259/GPerturb/blob/main/demo/TCells_GPerturb.ipynb)

Codes for reproducing the SciPlex2 example: [Link](https://github.com/hwxing3259/GPerturb/blob/main/demo/SciPlex2_GPerturb.ipynb)

Codes for reproducing the Replogle et al 2022 example: [Link](https://github.com/hwxing3259/GPerturb/blob/main/demo/Replogle_GPerturb.ipynb)
