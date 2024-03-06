from BSAPR_model import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import copy

# a simple and computationally efficient way to realize the modified two-group BSAPR is to assume that the 
# base function m_p(other_cell_info, cell_group) takes the form m_p(other_cell_info, 0) = m_{0p}(other_cell_info)
# and m_p(other_cell_info, 1) = m_{1p}(other_cell_info). We implement this choice of parameterization here
# However, when the number of cell group is large, it is probably better to directly parameterize m_p(other_cell_info, 0)
# as a single NN.

# load data
torch.manual_seed(3141592)
# load data:
my_conditioner = pd.read_csv('TCells_data/TCells_perturbation_mat.csv')
my_conditioner = my_conditioner.drop('NonTarget', axis=1)  # TODO: or retaining it
cond_name = list(my_conditioner.columns)
my_conditioner = torch.tensor(my_conditioner.to_numpy() * 1.0, dtype=torch.float)

my_observation = pd.read_csv('TCells_data/TCells_dev_filtered_raw.csv')
print(my_observation.shape)
gene_name = list(my_observation.columns)
my_observation = torch.tensor(my_observation.to_numpy() * 1.0, dtype=torch.float)

gene_name_lookup = pd.read_csv('TCells_data/TCells_unfiltered_gene_lookup.csv', index_col=0)
gene_name_lookup = {gene_name_lookup.iloc[i].unfiltered_genes: gene_name_lookup.iloc[i].feature_name_lookup_unfiltered for i in range(gene_name_lookup.shape[0])}
gene_name = [gene_name_lookup[i] for i in gene_name]

my_cell_info = pd.read_csv('TCells_data/TCells_cell_information.csv', index_col=0)
my_cell_info = my_cell_info[['lib_size', 'umi_count', 'percent_mt', 'doner', 'stimulated']]
my_cell_info.umi_count = my_cell_info.umi_count/my_cell_info.lib_size
my_cell_info.lib_size = np.log(my_cell_info.lib_size)
cell_info_names = list(my_cell_info.columns)
my_cell_info = torch.tensor(my_cell_info.to_numpy() * 1.0, dtype=torch.float)

stimulate_idx = my_cell_info[:, 4] == 0.
my_cell_info = my_cell_info[:, :4]


# design the training process:
start = time.time()
output_dim = my_observation.shape[1]
hidden_node = 1000
hidden_layer_1 = 4
hidden_layer_2 = 4
conditioner_dim = my_conditioner.shape[1]
cell_info_dim = my_cell_info.shape[1]

lr_parametric = 2e-3  # 7e-4 works, 2e-3 kinda works  # 1500, 1e-3, 0.005, 40, 0.37
nu_1, nu_2, nu_3, nu_4, nu_5, nu_6 = torch.tensor([1., 0.1, 1., 0.1, 1., 0.1]).to(device) # mean=-3, var=10
tau = torch.tensor(1.).to(device)
KL_factor = torch.tensor(1.).to(device)

# generate synthetic data
torch.manual_seed(314159)

my_observation_0, my_cell_info_0, my_conditioner_0 = my_observation[stimulate_idx], my_cell_info[stimulate_idx], my_conditioner[stimulate_idx]
my_dataset = MyDataSet(observation=my_observation_0, conditioner=my_conditioner_0, cell_info=my_cell_info_0)
N = my_observation_0.shape[0]
print(N)
testing_idx = set(np.random.choice(a=range(N), size=N//8, replace=False))
training_idx = list(set(range(N)) - testing_idx)
testing_idx = list(testing_idx)
training_idx_sampler = torch.utils.data.SubsetRandomSampler(training_idx)
training_loader = torch.utils.data.DataLoader(my_dataset, batch_size=100, sampler=training_idx_sampler)
test_observation = my_observation_0[testing_idx].to(device)
test_conditioner = my_conditioner_0[testing_idx].to(device)
test_cell_info = my_cell_info_0[testing_idx].to(device)

parametric_model = BSAPR_Gaussian(conditioner_dim=conditioner_dim, output_dim=output_dim, base_dim=cell_info_dim,
                                   data_size = N, hidden_node=hidden_node, hidden_layer_1=hidden_layer_1,
                                   hidden_layer_2=hidden_layer_2, tau=tau)
parametric_model = parametric_model.to(device)

optimizer = torch.optim.Adam(parametric_model.parameters(), lr=lr_parametric)


# training starts here
epoch = 250
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
                                                           cell_info=test_cell_info, nu_1=nu_1, nu_2=nu_2, nu_3=nu_3,
                                                           nu_4=nu_4, nu_5=nu_5, nu_6=nu_6, KL_factor=KL_factor,
                                                           test=True).detach().cpu().item()
        print('EPOCH={}, test_error={}'.format(EPOCH, testing_loss[EPOCH]))
    print('EPOCH={}, training_error={}'.format(EPOCH, training_loss[EPOCH]))
    # sch.step()
end = time.time()
print(end-start)


torch.save(parametric_model.state_dict(), 'BSAPR_TCells_Gaussian_cond0.pt')
np.savetxt(fname='BSAPR_TCells_Gaussian_training_loss_cond0.txt', X=training_loss.numpy())
np.savetxt(fname='BSAPR_TCells_Gaussian_testing_loss_cond0.txt', X=testing_loss.numpy())

# parametric_model.load_state_dict(torch.load('BSAPR_TCells_Gaussian2.pt'))
my_plot_Gaussian(model=parametric_model, obs=test_observation, cond=test_conditioner,
            cell_info=test_cell_info, full_cond=my_conditioner_0,
            gene_name=gene_name, cond_name=cond_name, x=-20, y=30, p=0.9,
            fig1='BSAPR_TCells_Gaussian_fig1_cond0.png',
            fig2='BSAPR_TCells_Gaussian_fig2_cond0.png')





# generate synthetic data
torch.manual_seed(314159)

my_observation_1, my_cell_info_1, my_conditioner_1 = my_observation[~stimulate_idx], my_cell_info[~stimulate_idx], my_conditioner[~stimulate_idx]
my_dataset = MyDataSet(observation=my_observation_1, conditioner=my_conditioner_1, cell_info=my_cell_info_1)
N = my_observation_1.shape[0]
print(N)
testing_idx = set(np.random.choice(a=range(N), size=N//8, replace=False))
training_idx = list(set(range(N)) - testing_idx)
testing_idx = list(testing_idx)
training_idx_sampler = torch.utils.data.SubsetRandomSampler(training_idx)
training_loader = torch.utils.data.DataLoader(my_dataset, batch_size=100, sampler=training_idx_sampler)
test_observation = my_observation_1[testing_idx].to(device)
test_conditioner = my_conditioner_1[testing_idx].to(device)
test_cell_info = my_cell_info_1[testing_idx].to(device)

parametric_model = BSAPR_Gaussian(conditioner_dim=conditioner_dim, output_dim=output_dim, base_dim=cell_info_dim,
                                   data_size = N, hidden_node=hidden_node, hidden_layer_1=hidden_layer_1,
                                   hidden_layer_2=hidden_layer_2, tau=tau)
parametric_model = parametric_model.to(device)

optimizer = torch.optim.Adam(parametric_model.parameters(), lr=lr_parametric)


# training starts here
epoch = 250
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
                                                           cell_info=test_cell_info, nu_1=nu_1, nu_2=nu_2, nu_3=nu_3,
                                                           nu_4=nu_4, nu_5=nu_5, nu_6=nu_6, KL_factor=KL_factor,
                                                           test=True).detach().cpu().item()
        print('EPOCH={}, test_error={}'.format(EPOCH, testing_loss[EPOCH]))
    print('EPOCH={}, training_error={}'.format(EPOCH, training_loss[EPOCH]))
    # sch.step()
end = time.time()
print(end-start)


torch.save(parametric_model.state_dict(), 'BSAPR_TCells_Gaussian_cond1.pt')
np.savetxt(fname='BSAPR_TCells_Gaussian_training_loss_cond1.txt', X=training_loss.numpy())
np.savetxt(fname='BSAPR_TCells_Gaussian_testing_loss_cond1.txt', X=testing_loss.numpy())

# parametric_model.load_state_dict(torch.load('BSAPR_TCells_Gaussian2.pt'))
my_plot_Gaussian(model=parametric_model, obs=test_observation, cond=test_conditioner,
            cell_info=test_cell_info, full_cond=my_conditioner,
            gene_name=gene_name, cond_name=cond_name, x=-20, y=30, p=0.9,
            fig1='BSAPR_TCells_Gaussian_fig1_cond1.png',
            fig2='BSAPR_TCells_Gaussian_fig2_cond1.png')






#
# # design the training process:
# start = time.time()
# output_dim = my_observation.shape[1]
# sample_size = my_observation.shape[0]
# hidden_node = 1000
# hidden_layer_1 = 4
# hidden_layer_2 = 4
# conditioner_dim = my_conditioner.shape[1]
# cell_info_dim = my_cell_info.shape[1]
#
# lr_parametric = 1e-3  # 7e-4 works, 2e-3 kinda works  # 1500, 1e-3, 0.005, 40, 0.37
# nu_1, nu_2, nu_3, nu_4, nu_5, nu_6 = torch.tensor([1., 0.1, 1., 0.1, 1., 0.1]).to(device) # mean=-3, var=10
# tau = torch.tensor(1.).to(device)
# KL_factor = torch.tensor(1.).to(device)
#
# # generate synthetic data
# torch.manual_seed(314159)
#
# my_dataset = MyDataSet(observation=my_observation, conditioner=my_conditioner, cell_info=my_cell_info)
#
# testing_idx = set(np.random.choice(a=range(my_observation.shape[0]), size=my_observation.shape[0]//8, replace=False))
# training_idx = list(set(range(my_observation.shape[0])) - testing_idx)
# testing_idx = list(testing_idx)
# training_idx_sampler = torch.utils.data.SubsetRandomSampler(training_idx)
# training_loader = torch.utils.data.DataLoader(my_dataset, batch_size=100, sampler=training_idx_sampler)
# test_observation = my_observation[testing_idx].to(device)
# test_conditioner = my_conditioner[testing_idx].to(device)
# test_cell_info = my_cell_info[testing_idx].to(device)
#
# parametric_model = BSAPR_Gaussian2(conditioner_dim=conditioner_dim, output_dim=output_dim, base_dim=cell_info_dim,
#                                    data_size = sample_size, hidden_node=hidden_node, hidden_layer_1=hidden_layer_1,
#                                    hidden_layer_2=hidden_layer_2, tau=tau)
# parametric_model = parametric_model.to(device)
#
# optimizer = torch.optim.Adam(parametric_model.parameters(), lr=lr_parametric)
#
#
# # training starts here
# epoch = 250
# training_loss = torch.zeros(epoch)
# testing_loss = torch.zeros(epoch)
# for EPOCH in range(epoch):
#     curr_training_loss = 0.
#     parametric_model.train()
#     for i, (obs, cond, cell_info) in enumerate(training_loader):
#         obs = obs.to(device)
#         cond = cond.to(device)
#         cell_info = cell_info.to(device)
#         loss = parametric_model.normal_loss(observation=obs, conditioner=cond, cell_info=cell_info,
#                                             nu_1=nu_1, nu_2=nu_2, nu_3=nu_3, nu_4=nu_4, nu_5=nu_5, nu_6=nu_6,
#                                             KL_factor=KL_factor, additional_dim=additional_dim)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         curr_training_loss += loss.detach().cpu().item() * obs.shape[0]
#     training_loss[EPOCH] = curr_training_loss/len(training_idx)
#     parametric_model.eval()
#     with torch.no_grad():
#         testing_loss[EPOCH] = parametric_model.normal_loss(observation=test_observation, conditioner=test_conditioner,
#                                                            cell_info=test_cell_info, nu_1=nu_1, nu_2=nu_2, nu_3=nu_3,
#                                                            nu_4=nu_4, nu_5=nu_5, nu_6=nu_6, KL_factor=KL_factor,
#                                                            test=True, additional_dim=additional_dim).detach().cpu().item()
#         print('EPOCH={}, test_error={}'.format(EPOCH, testing_loss[EPOCH]))
#     print('EPOCH={}, training_error={}'.format(EPOCH, training_loss[EPOCH]))
#     # sch.step()
# end = time.time()
# print(end-start)
#
#
# torch.save(parametric_model.state_dict(), 'BSAPR_TCells_Gaussian2.pt')
# np.savetxt(fname='BSAPR_TCells_Gaussian_training_loss2.txt', X=training_loss.numpy())
# np.savetxt(fname='BSAPR_TCells_Gaussian_testing_loss2.txt', X=testing_loss.numpy())
#
# # parametric_model.load_state_dict(torch.load('BSAPR_TCells_Gaussian2.pt'))
# my_plot_Gaussian(model=parametric_model, obs=test_observation, cond=test_conditioner,
#             cell_info=test_cell_info, full_cond=my_conditioner,
#             gene_name=gene_name, cond_name=cond_name, x=-20, y=30, p=0.9,
#             fig1='BSAPR_TCells_Gaussian_fig12.png',
#             fig2='BSAPR_TCells_Gaussian_fig22.png', additional_dim=additional_dim)






my_conditioner = pd.read_csv('TCells_data/TCells_perturbation_mat.csv')
my_conditioner = my_conditioner.drop('NonTarget', axis=1)  # TODO: or retaining it
cond_name = list(my_conditioner.columns)
my_conditioner = torch.tensor(my_conditioner.to_numpy() * 1.0, dtype=torch.float)

my_observation = pd.read_csv('TCells_data/TCells_count_unfiltered_raw.csv', index_col=0)
print(my_observation.shape)
gene_name = list(my_observation.columns)
my_observation = torch.tensor(my_observation.to_numpy() * 1.0, dtype=torch.float)

gene_name_lookup = pd.read_csv('TCells_data/TCells_unfiltered_gene_lookup.csv', index_col=0)
gene_name_lookup = {gene_name_lookup.iloc[i].unfiltered_genes: gene_name_lookup.iloc[i].feature_name_lookup_unfiltered for i in range(gene_name_lookup.shape[0])}
gene_name = [gene_name_lookup[i] for i in gene_name]

my_cell_info = pd.read_csv('TCells_data/TCells_cell_information.csv', index_col=0)
my_cell_info = my_cell_info[['lib_size', 'umi_count', 'percent_mt', 'doner', 'stimulated']]
my_cell_info.umi_count = my_cell_info.umi_count/my_cell_info.lib_size
my_cell_info.lib_size = np.log(my_cell_info.lib_size)
cell_info_names = list(my_cell_info.columns)
my_cell_info = torch.tensor(my_cell_info.to_numpy() * 1.0, dtype=torch.float)

stimulate_idx = my_cell_info[:, 4] == 0.
my_cell_info = my_cell_info[:, :4]

# design the training process:
start = time.time()
output_dim = my_observation.shape[1]
hidden_node =1000
hidden_layer_1 = 4
hidden_layer_2 = 4
conditioner_dim = my_conditioner.shape[1]
cell_info_dim = my_cell_info.shape[1]

lr_parametric = 3e-4  # 7e-4 works, 2e-3 kinda works
nu_1, nu_2, nu_3, nu_4 = torch.tensor(3.).to(device), torch.tensor(0.1).to(device), torch.tensor(3.).to(device), torch.tensor(0.1).to(device)
tau = torch.tensor(1.).to(device)
KL_factor = torch.tensor(1.).to(device)

# generate synthetic data
torch.manual_seed(314159)
np.random.seed(314159)
my_observation_0, my_cell_info_0, my_conditioner_0 = my_observation[stimulate_idx], my_cell_info[stimulate_idx], my_conditioner[stimulate_idx]
my_dataset = MyDataSet(observation=my_observation_0, conditioner=my_conditioner_0, cell_info=my_cell_info_0)
N = my_observation_0.shape[0]
testing_idx = set(np.random.choice(a=range(my_observation_0.shape[0]), size=my_observation_0.shape[0]//8, replace=False))
training_idx = list(set(range(my_observation_0.shape[0])) - testing_idx)
testing_idx = list(testing_idx)
training_idx_sampler = torch.utils.data.SubsetRandomSampler(training_idx)
training_loader = torch.utils.data.DataLoader(my_dataset, batch_size=256, sampler=training_idx_sampler)
test_observation = my_observation_0[testing_idx].to(device)
test_conditioner = my_conditioner_0[testing_idx].to(device)
test_cell_info = my_cell_info_0[testing_idx].to(device)

parametric_model = BSAPR_ZIP(conditioner_dim=conditioner_dim, output_dim=output_dim,
                                    base_dim=cell_info_dim, data_size=N,
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

torch.save(parametric_model.state_dict(), 'BSAPR_TCells_ZIP_cond0.pt')

est_0 = my_plot_ZIP(model=parametric_model, obs=test_observation, cond=test_conditioner,
            cell_info=test_cell_info, full_cond=my_conditioner,
            gene_name=gene_name, cond_name=cond_name, x=200, y=800,
            fig1='BSAPR_TCells_ZIP_fig1_cond00.png',
            fig2='BSAPR_TCells_ZIP_fig2_cond00.png')






parametric_model.load_state_dict(torch.load('BSAPR_TCells_ZIP_cond0.pt'))

threshold = 0.95
top = 50
unique_conditions = torch.unique(my_conditioner_0, dim=0).to(device)
perturb_level, _, _, logit_p, _ = parametric_model(unique_conditions, None)
estimated_inclusion_prob = F.sigmoid(logit_p).detach().cpu().numpy()
estimated_inclusion = estimated_inclusion_prob > threshold
my_gene_name = np.array(gene_name)[estimated_inclusion.sum(axis=0) > 0]
estimated_inclusion_prob = estimated_inclusion_prob[:, estimated_inclusion.mean(axis=0) > 0]
estimated_pert = perturb_level.detach().cpu().numpy() * estimated_inclusion
estimated_pert = estimated_pert[:, estimated_inclusion.mean(axis=0) > 0]
top_gene = np.argsort((estimated_pert**2).sum(axis=0))[::-1][:top]
estimated_pert = estimated_pert[:, top_gene]

unique_conditions = unique_conditions.cpu().numpy()
my_yticks = ['' for _ in range(unique_conditions.shape[0])]
for i in range(unique_conditions.shape[0]):
    if np.all(unique_conditions[i] == 0):
        my_yticks[i] = 'Non Targeting'
    else:
        my_yticks[i] = np.array(cond_name)[unique_conditions[i] == 1][0] + '_N'


estimated_pert_0 = estimated_pert[1:, :]*1.0
my_gene_name_0 = copy.copy(my_gene_name[top_gene])
my_yticks_0 = copy.copy(my_yticks[1:])

pert_mean = np.zeros((my_observation.shape[0], output_dim))


predicted_mu_mean, _, predicted_base_mean, logit_p, _ = parametric_model(my_conditioner_0.to(device), my_cell_info_0.to(device))
estimated_base_mean = predicted_base_mean.detach().cpu().numpy()  # * zeros[testing_idx].numpy()
estimated_perturbed_mean = (F.sigmoid(logit_p) * predicted_mu_mean).detach().cpu().numpy()
estimated_perturbed_rate = (logexpp1(torch.tensor(estimated_base_mean + estimated_perturbed_mean))).numpy()

pert_mean[stimulate_idx] = estimated_perturbed_rate








my_observation_1, my_cell_info_1, my_conditioner_1 = my_observation[~stimulate_idx], my_cell_info[~stimulate_idx], my_conditioner[~stimulate_idx]
my_dataset = MyDataSet(observation=my_observation_1, conditioner=my_conditioner_1, cell_info=my_cell_info_1)
N = my_observation_1.shape[0]
testing_idx = set(np.random.choice(a=range(my_observation_1.shape[0]), size=my_observation_1.shape[0]//8, replace=False))
training_idx = list(set(range(my_observation_1.shape[0])) - testing_idx)
testing_idx = list(testing_idx)
training_idx_sampler = torch.utils.data.SubsetRandomSampler(training_idx)
training_loader = torch.utils.data.DataLoader(my_dataset, batch_size=256, sampler=training_idx_sampler)
test_observation = my_observation_1[testing_idx].to(device)
test_conditioner = my_conditioner_1[testing_idx].to(device)
test_cell_info = my_cell_info_1[testing_idx].to(device)

parametric_model = BSAPR_ZIP(conditioner_dim=conditioner_dim, output_dim=output_dim,
                                    base_dim=cell_info_dim, data_size=N,
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

torch.save(parametric_model.state_dict(), 'BSAPR_TCells_ZIP_cond1.pt')

est_1 = my_plot_ZIP(model=parametric_model, obs=test_observation, cond=test_conditioner,
            cell_info=test_cell_info, full_cond=my_conditioner,
            gene_name=gene_name, cond_name=cond_name, x=200, y=800,
            fig1='BSAPR_TCells_ZIP_fig1_cond1.png',
            fig2='BSAPR_TCells_ZIP_fig2_cond1.png')













