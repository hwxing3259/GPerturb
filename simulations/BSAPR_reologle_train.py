from BSAPR_model import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import copy
import scanpy as sc

adata = sc.read_h5ad('replogle.h5ad')
adata.obs['n_feature'] = (adata.X > 0).sum(1)

# zip
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



# design the training process:
start = time.time()
output_dim = my_observation.shape[1]
sample_size = my_observation.shape[0]
hidden_node = 1000
hidden_layer_1 = 4
hidden_layer_2 = 4
conditioner_dim = my_conditioner.shape[1]
cell_info_dim = my_cell_info.shape[1]

lr_parametric = 5e-4  # 7e-4 works, 2e-3 kinda works
nu_1, nu_2, nu_3, nu_4 = torch.tensor(5.).to(device), torch.tensor(0.1).to(device), torch.tensor(1.).to(device), torch.tensor(0.1).to(device)
tau = torch.tensor(1.).to(device)
KL_factor = torch.tensor(1.).to(device)

# generate synthetic data
torch.manual_seed(314159)

my_dataset = MyDataSet(observation=my_observation, conditioner=my_conditioner, cell_info=my_cell_info)

testing_idx = set(np.random.choice(a=range(my_observation.shape[0]), size=my_observation.shape[0]//8, replace=False))
training_idx = list(set(range(my_observation.shape[0])) - testing_idx)
testing_idx = list(testing_idx)
training_idx_sampler = torch.utils.data.SubsetRandomSampler(training_idx)
training_loader = torch.utils.data.DataLoader(my_dataset, batch_size=100, sampler=training_idx_sampler)
test_observation = my_observation[testing_idx].to(device)
test_conditioner = my_conditioner[testing_idx].to(device)
test_cell_info = my_cell_info[testing_idx].to(device)

parametric_model = BSAPR_ZIP(conditioner_dim=conditioner_dim, output_dim=output_dim,
                             base_dim=cell_info_dim, data_size=sample_size,
                             hidden_node=hidden_node, hidden_layer_1=hidden_layer_1,
                             hidden_layer_2=hidden_layer_2, tau=tau)
parametric_model = parametric_model.to(device)

optimizer = torch.optim.Adam(parametric_model.parameters(), lr=lr_parametric)


# training starts here
epoch = 300
training_loss = torch.zeros(epoch)
testing_loss = torch.zeros(epoch)
for EPOCH in range(epoch):
    curr_training_loss = 0.
    for i, (obs, cond, cell_info) in enumerate(training_loader):
        obs = obs.to(device)
        cond = cond.to(device)
        cell_info = cell_info.to(device)
        loss = parametric_model.zip_loss(observation=obs, conditioner=cond, cell_info=cell_info,
                                         nu_1=nu_1, nu_2=nu_2, nu_3=nu_3, nu_4=nu_4,
                                         KL_factor=KL_factor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        curr_training_loss += loss.detach().cpu().item() * obs.shape[0]
    training_loss[EPOCH] = curr_training_loss/len(training_idx)
    with torch.no_grad():
        testing_loss[EPOCH] = parametric_model.zip_loss(observation=test_observation,
                                                        conditioner=test_conditioner,
                                                        cell_info=test_cell_info,
                                                        nu_1=nu_1, nu_2=nu_2, nu_3=nu_3, nu_4=nu_4,
                                                        KL_factor=KL_factor,
                                                        test=True).detach().cpu().item()
        print('EPOCH={}, test_error={}'.format(EPOCH, testing_loss[EPOCH]))
    print('EPOCH={}, training_error={}'.format(EPOCH, training_loss[EPOCH]))
    # sch.step()
end = time.time()
print(end-start)

torch.save(parametric_model.state_dict(), 'BSAPR_reo_ZIP_ref.pt')
np.savetxt(fname='BSAPR_reo_ZIP_training_loss_ref.txt', X=training_loss.numpy())
np.savetxt(fname='BSAPR_reo_ZIP_testing_loss_ref.txt', X=testing_loss.numpy())

my_plot_ZIP(model=parametric_model, obs=test_observation, cond=test_conditioner,
            cell_info=test_cell_info, full_cond=my_conditioner,
            gene_name=gene_name, cond_name=cond_name, x=200, y=1500,
            fig1='BSAPR_reo_ZIP_fig1_ref_2.png',
            fig2='BSAPR_reo_ZIP_fig2_ref_2.png')




device='cpu'
parametric_model = BSAPR_ZIP(conditioner_dim=conditioner_dim, output_dim=output_dim, base_dim=cell_info_dim,
                               data_size=sample_size, hidden_node=hidden_node, hidden_layer_1=hidden_layer_1,
                               hidden_layer_2=hidden_layer_2, tau=tau)
parametric_model.load_state_dict(torch.load('BSAPR_reo_ZIP_ref.pt'))
unique_pert = adata.obs.gene.unique()
avg_pred = np.zeros((len(unique_pert), adata.X.shape[1]))
avg_obs = np.zeros((len(unique_pert), adata.X.shape[1]))
for i, pert in enumerate(list(unique_pert)):
    if i%10 == 0:
        print(i)
    my_id = pert == adata.obs.gene
    predicted_mu_mean, predicted_mu_var, predicted_base_mean, logit_p, logit_p_log_var = parametric_model(my_conditioner[my_id], my_cell_info[my_id])
    estimated_base_mean = predicted_base_mean  # * zeros[testing_idx].numpy()
    estimated_perturbed_mean = (F.sigmoid(logit_p) * predicted_mu_mean)
    avg_pred[i] = logexpp1(estimated_perturbed_mean + estimated_base_mean).detach().cpu().numpy().mean(0)
    avg_obs[i] = my_observation[my_id].mean(0).numpy()

np.savetxt('reo_avg_pred_BSAPR_zip.csv', avg_pred)
np.savetxt('reo_avg_obs_BSAPR_zip.csv', avg_obs)

fig, axes = plt.subplots(1, 1)
axes.scatter(avg_pred[avg_obs != 0], avg_obs[avg_obs != 0], alpha=0.15)
axes.axline((1, 1), slope=1, c='r', alpha=0.5, linestyle='--')
axes.set_xlabel('Averaged prediction for each unique perturbation')
axes.set_ylabel('Averaged observation for each unique perturbation')
axes.set_title('Reologle, Predicted vs Observed')
fig.set_size_inches(12, 12)
plt.savefig('reo_ZIP.png')
plt.close()


threshold = 0.95
top=200
unique_conditions = torch.unique(my_conditioner, dim=0)
perturb_level, _, _, logit_p, _ = parametric_model(unique_conditions, None)
estimated_inclusion_prob = F.sigmoid(logit_p).detach().cpu().numpy()
estimated_inclusion = estimated_inclusion_prob > threshold
my_gene_name = np.array(gene_name)[estimated_inclusion.sum(axis=0) > 0]
estimated_inclusion_prob = estimated_inclusion_prob[:, estimated_inclusion.mean(axis=0) > 0]
estimated_pert = perturb_level.detach().cpu().numpy() * estimated_inclusion
estimated_pert = estimated_pert[:, estimated_inclusion.mean(axis=0) > 0]


unique_conditions = unique_conditions.numpy()
my_yticks = ['' for _ in range(unique_conditions.shape[0])]
ref_id = 0
for i in range(unique_conditions.shape[0]):
    if np.all(unique_conditions[i] == 0):
        my_yticks[i] = 'Non Targeting'
        ref_id = i
    else:
        my_yticks[i] = np.array(cond_name)[unique_conditions[i] == 1][0]

pathway_id = {}

for path in pathways:
    pathway_id[path] = [my_yticks.index(i) for i in pathways[path]]


my_yticks_reo = np.array(my_yticks)[np.concatenate(list(pathway_id.values()))]
estimated_pert_reo = estimated_pert[np.concatenate(list(pathway_id.values()))]
top_gene = np.argsort((estimated_pert**2).sum(axis=0))[::-1][:top]
estimated_pert_reo = estimated_pert_reo[:, top_gene]
my_gene_name_reo = my_gene_name[top_gene]

import matplotlib.colors as colors

fig2, axes2 = plt.subplots(1, 1)
# plt.set_cmap('RdBu')
# negatives = estimated_pert_reo.min()
# positives = estimated_pert_reo.max()
#
# bounds_min = np.linspace(negatives, 0, 129)
# bounds_max = np.linspace(0, positives, 129)[1:]
#     # the zero is only needed once
#     # in total there will be 257 bounds, so 256 bins
# bounds = np.concatenate((bounds_min, bounds_max), axis=None)
# norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
#
# x, y = np.meshgrid(np.arange(estimated_pert_LUHMES.shape[1]), np.arange(estimated_pert_LUHMES.shape[0]))
# im = axes2[0].pcolormesh(x,y,estimated_pert_LUHMES, norm=norm, cmap="RdBu_r")


negatives = estimated_pert_reo.min()
positives = estimated_pert_reo.max()
num_neg_colors = int(256 / (positives - negatives) * (-negatives))
num_pos_colors = 256 - num_neg_colors
cmap_BuRd = plt.cm.RdBu_r
colors_2neg_4pos = [cmap_BuRd(0.5*c/num_neg_colors) for c in range(num_neg_colors)] +\
                   [cmap_BuRd(1-0.5*c/num_pos_colors) for c in range(num_pos_colors)][::-1]
cmap_2neg_4pos = colors.LinearSegmentedColormap.from_list('cmap_2neg_4pos', colors_2neg_4pos, N=256)

im = axes2.imshow(estimated_pert_reo, cmap=cmap_2neg_4pos)
# im = axes2.imshow(estimated_pert_LUHMES, norm=norm, cmap='RdBu_r')
# axes2.set_xticks(np.arange(len(my_gene_name_reo)), my_gene_name_reo, rotation=90)
# axes2.set_yticks(np.arange(len(my_yticks_reo)), my_yticks_reo)
axes2.set_xticks([])
axes2.set_yticks([])
axes2.set_ylabel('')
for l in np.cumsum([len(i) for i in pathways.values()]):
    axes2.axline((0., l-0.5), slope=0, alpha=0.4, c='k')
axes2.set_xlabel('Perturbed Genes')
axes2.set_title('Estimated perturbation effects')
axes2.set_aspect('equal')
# ticks = np.append(np.arange(-1.5, 0, 0.5), np.arange(0, 5.101, 1.7))
# fig2.colorbar(im, ax=axes2, ticks=ticks)
fig2.colorbar(im, ax=axes2)
fig2.set_size_inches(8, 12)
fig2.tight_layout()
plt.savefig('reo_BSAPR_heatmap_ZIP.png')
plt.close()











# Gaussian
adata = sc.read_h5ad('replogle.h5ad')
adata.obs['n_feature'] = (adata.X > 0).sum(1)
my_observation = adata.X / adata.obs.core_scale_factor.values[:, None]
gene_name = list(adata.var.gene_name)
my_observation = torch.tensor(np.log(my_observation + 1.), dtype=torch.float)

my_cell_info = adata.obs[['core_adjusted_UMI_count', 'mitopercent', 'n_feature', 'core_scale_factor']]
my_cell_info = torch.tensor(my_cell_info.to_numpy() * 1.0, dtype=torch.float)
my_cell_info[:, 2] = my_cell_info[:, 2] / my_cell_info[:, 0]
my_cell_info[:, 0] = np.log(my_cell_info[:, 0])

pathways = adata.uns['pathways']

my_conditioner = pd.get_dummies(adata.obs['gene'])
my_conditioner = my_conditioner.drop('non-targeting', axis=1)
cond_name = list(my_conditioner.columns)
my_conditioner = torch.tensor(my_conditioner.to_numpy() * 1.0, dtype=torch.float)


# design the training process:
start = time.time()
output_dim = my_observation.shape[1]
sample_size = my_observation.shape[0]
hidden_node = 1000  # or 1000
hidden_layer_1 = 4
hidden_layer_2 = 4
conditioner_dim = my_conditioner.shape[1]
cell_info_dim = my_cell_info.shape[1]

lr_parametric = 1e-3  # 7e-4 works, 2e-3 kinda works  # 1500, 1e-3, 0.005, 40, 0.37
nu_1, nu_2, nu_3, nu_4, nu_5, nu_6 = torch.tensor([1., 0.1, 1., 0.1, 1., 0.1]).to(device)
tau = torch.tensor(1.0).to(device)
KL_factor = torch.tensor(1.).to(device)

# generate synthetic data
torch.manual_seed(314159)

my_dataset = MyDataSet(observation=my_observation, conditioner=my_conditioner, cell_info=my_cell_info)

testing_idx = set(np.random.choice(a=range(my_observation.shape[0]), size=my_observation.shape[0]//8, replace=False))
training_idx = list(set(range(my_observation.shape[0])) - testing_idx)
testing_idx = list(testing_idx)
training_idx_sampler = torch.utils.data.SubsetRandomSampler(training_idx)
training_loader = torch.utils.data.DataLoader(my_dataset, batch_size=100, sampler=training_idx_sampler)
test_observation = my_observation[testing_idx].to(device)
test_conditioner = my_conditioner[testing_idx].to(device)
test_cell_info = my_cell_info[testing_idx].to(device)

parametric_model = BSAPR_Gaussian(conditioner_dim=conditioner_dim, output_dim=output_dim, base_dim=cell_info_dim,
                               data_size=sample_size, hidden_node=hidden_node, hidden_layer_1=hidden_layer_1,
                               hidden_layer_2=hidden_layer_2, tau=tau)
parametric_model = parametric_model.to(device)

optimizer = torch.optim.Adam(parametric_model.parameters(), lr=lr_parametric)


# training starts here
epoch = 300
training_loss = torch.zeros(epoch)
testing_loss = torch.zeros(epoch)
for EPOCH in range(epoch):
    curr_training_loss = 0.
    parametric_model.train()
    for i, (obs, cond, cell_info) in enumerate(training_loader):
        obs = obs.to(device)
        cond = cond.to(device)
        cell_info = cell_info.to(device)
        loss = parametric_model.normal_loss(observation=obs, conditioner=cond, cell_info=cell_info,
                                            nu_1=nu_1, nu_2=nu_2, nu_3=nu_3, nu_4=nu_4, nu_5=nu_5, nu_6=nu_6,
                                            KL_factor=KL_factor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        curr_training_loss += loss.detach().cpu().item() * obs.shape[0]
    training_loss[EPOCH] = curr_training_loss/len(training_idx)
    parametric_model.eval()
    with torch.no_grad():
        testing_loss[EPOCH] = parametric_model.normal_loss(observation=test_observation, conditioner=test_conditioner,
                                                           cell_info=test_cell_info, nu_1=nu_1, nu_2=nu_2,
                                                           nu_3=nu_3, nu_4=nu_4,nu_5=nu_5, nu_6=nu_6,
                                                           KL_factor=KL_factor, test=True).detach().cpu().item()
        print('EPOCH={}, test_error={}'.format(EPOCH, testing_loss[EPOCH]))
    print('EPOCH={}, training_error={}'.format(EPOCH, training_loss[EPOCH]))
    # sch.step()
end = time.time()
print(end-start)


torch.save(parametric_model.state_dict(), 'BSAPR_reo_Gaussian_ref.pt')
np.savetxt(fname='BSAPR_reo_Gaussian_training_loss_ref.txt', X=training_loss.numpy())
np.savetxt(fname='BSAPR_reo_Gaussian_testing_loss_ref.txt', X=testing_loss.numpy())

my_plot_Gaussian(model=parametric_model, obs=test_observation, cond=test_conditioner,
                 cell_info=test_cell_info, full_cond=my_conditioner,
                 gene_name=gene_name, cond_name=cond_name, x=-20, y=30,
                 fig1='BSAPR_reo_Gaussian_fig1_ref_2.png',
                 fig2='BSAPR_reo_Gaussian_fig2_ref_2.png')

device='cpu'
parametric_model = BSAPR_Gaussian(conditioner_dim=conditioner_dim, output_dim=output_dim, base_dim=cell_info_dim,
                               data_size=sample_size, hidden_node=hidden_node, hidden_layer_1=hidden_layer_1,
                               hidden_layer_2=hidden_layer_2, tau=tau)
parametric_model.load_state_dict(torch.load('BSAPR_reo_Gaussian_ref.pt'))
unique_pert = adata.obs.gene.unique()
avg_pred = np.zeros((len(unique_pert), adata.X.shape[1]))
avg_obs = np.zeros((len(unique_pert), adata.X.shape[1]))
for i, pert in enumerate(list(unique_pert)):
    if i%10 == 0:
        print(i)
    my_id = pert == adata.obs.gene
    predicted_mu_mean, predicted_mu_var, predicted_gamma_mean, predicted_gamma_var, \
        logit_p, logit_p_log_var, predicted_base_mean = parametric_model(my_conditioner[my_id],
                                                                         my_cell_info[my_id])
    estimated_base_mean = predicted_base_mean.detach().cpu().numpy()  # * zeros[testing_idx].numpy()
    estimated_perturbed_mean = (F.sigmoid(logit_p) * predicted_mu_mean).detach().cpu().numpy()
    avg_pred[i] = (estimated_perturbed_mean + estimated_base_mean).mean(0)
    avg_obs[i] = my_observation[my_id].mean(0).numpy()

np.savetxt('reo_avg_pred_BSAPR.csv', avg_pred)
np.savetxt('reo_avg_obs_BSAPR.csv', avg_obs)

fig, axes = plt.subplots(1, 1)
axes.scatter(avg_pred[avg_obs != 0], avg_obs[avg_obs != 0], alpha=0.15)
axes.axline((1, 1), slope=1, c='r', alpha=0.5, linestyle='--')
axes.set_xlabel('Averaged prediction for each unique perturbation')
axes.set_ylabel('Averaged observation for each unique perturbation')
axes.set_title('Reologle, Predicted vs Observed')
fig.set_size_inches(12, 12)
plt.savefig('reo_Gaussian.png')
plt.close()


threshold = 0.95
top=200
unique_conditions = torch.unique(my_conditioner, dim=0)
perturb_level, _, _, _, logit_p, _, _ = parametric_model(unique_conditions, None)
estimated_inclusion_prob = F.sigmoid(logit_p).detach().cpu().numpy()
estimated_inclusion = estimated_inclusion_prob > threshold
my_gene_name = np.array(gene_name)[estimated_inclusion.sum(axis=0) > 0]
estimated_inclusion_prob = estimated_inclusion_prob[:, estimated_inclusion.mean(axis=0) > 0]
estimated_pert = perturb_level.detach().cpu().numpy() * estimated_inclusion
estimated_pert = estimated_pert[:, estimated_inclusion.mean(axis=0) > 0]


unique_conditions = unique_conditions.numpy()
my_yticks = ['' for _ in range(unique_conditions.shape[0])]
ref_id = 0
for i in range(unique_conditions.shape[0]):
    if np.all(unique_conditions[i] == 0):
        my_yticks[i] = 'Non Targeting'
        ref_id = i
    else:
        my_yticks[i] = np.array(cond_name)[unique_conditions[i] == 1][0]

pathway_id = {}

for path in pathways:
    pathway_id[path] = [my_yticks.index(i) for i in pathways[path]]


my_yticks_reo = np.array(my_yticks)[np.concatenate(list(pathway_id.values()))]
estimated_pert_reo = estimated_pert[np.concatenate(list(pathway_id.values()))]
top_gene = np.argsort((estimated_pert**2).sum(axis=0))[::-1][:top]
estimated_pert_reo = estimated_pert_reo[:, top_gene]
my_gene_name_reo = my_gene_name[top_gene]

import matplotlib.colors as colors

fig2, axes2 = plt.subplots(1, 1)
# plt.set_cmap('RdBu')
negatives = estimated_pert_reo.min()
positives = estimated_pert_reo.max()

bounds_min = np.linspace(negatives, 0, 129)
bounds_max = np.linspace(0, positives, 129)[1:]
    # the zero is only needed once
    # in total there will be 257 bounds, so 256 bins
bounds = np.concatenate((bounds_min, bounds_max), axis=None)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
#
# x, y = np.meshgrid(np.arange(estimated_pert_LUHMES.shape[1]), np.arange(estimated_pert_LUHMES.shape[0]))
# im = axes2[0].pcolormesh(x,y,estimated_pert_LUHMES, norm=norm, cmap="RdBu_r")


negatives = estimated_pert.min()
positives = estimated_pert.max()
num_neg_colors = int(256 / (positives - negatives) * (-negatives))
num_pos_colors = 256 - num_neg_colors
cmap_BuRd = plt.cm.RdBu_r
colors_2neg_4pos = [cmap_BuRd(0.5*c/num_neg_colors) for c in range(num_neg_colors)] +\
                   [cmap_BuRd(1-0.5*c/num_pos_colors) for c in range(num_pos_colors)][::-1]
cmap_2neg_4pos = colors.LinearSegmentedColormap.from_list('cmap_2neg_4pos', colors_2neg_4pos, N=256)

im = axes2.imshow(estimated_pert_reo, cmap=cmap_2neg_4pos)
# im = axes2.imshow(estimated_pert_LUHMES, norm=norm, cmap='RdBu_r')
# axes2.set_xticks(np.arange(len(my_gene_name_reo)), my_gene_name_reo, rotation=90)
# axes2.set_yticks(np.arange(len(my_yticks_reo)), my_yticks_reo)
axes2.set_xticks([])
axes2.set_yticks([])
axes2.set_ylabel('')
for l in np.cumsum([len(i) for i in pathways.values()]):
    axes2.axline((0., l-0.5), slope=0, alpha=0.4, c='k')
axes2.set_xlabel('Perturbed Genes')
axes2.set_title('Estimated perturbation effects')
axes2.set_aspect('equal')
# ticks = np.append(np.arange(-1.5, 0, 0.5), np.arange(0, 5.101, 1.7))
# fig2.colorbar(im, ax=axes2, ticks=ticks)
fig2.colorbar(im, ax=axes2)
fig2.set_size_inches(8, 12)
fig2.tight_layout()
plt.savefig('reo_BSAPR_heatmap.png')
plt.close()
