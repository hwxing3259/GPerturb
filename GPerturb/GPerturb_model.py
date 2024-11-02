import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# argh! boils down to numerical stability!
def logexpp1(x):
    ans = 1.0*x
    ans[x < 12.] = torch.log(torch.exp(x[x < 12.]) + 1.)+1e-4
    return ans


def loglogexpp1(x):
    ans = 1.0*x
    ans[x >= 12.] = torch.log(x[x >= 12.])
    ans[x < 12] = torch.log(torch.log(torch.exp(x[x < 12]) + 1.)+1e-4)
    return ans


class MyDataSet(torch.utils.data.Dataset):
    """
    q1, q1 are samples from q1,q2, stack them, give label 1 if it's from q1, 0 if it's from q2,
    don't actually need the weight, too lazy to remove it lol
    """
    def __init__(self, observation, conditioner, cell_info):
        super(MyDataSet, self).__init__()
        self.obs = observation
        self.conditioner = conditioner
        self.cell_info = cell_info

    def __getitem__(self, item):
        return self.obs[item], self.conditioner[item], self.cell_info[item]

    def __len__(self):  # returns n1+n2
        return self.obs.shape[0]


# variable prior variance across all genes should be equivalent to putting a scaled t prior on the perturbations
# fixing 0 mean is sensible, as the additive structure ensures that any non-zero mean will be absorbed into baseline
# should the hyper parameters alpha,betas be variable?

class GPerturb_Gaussian(nn.Module):
    def __init__(self, conditioner_dim, output_dim, base_dim, data_size, hidden_node, hidden_layer_1, hidden_layer_2, tau):
        # base_dim consists 4 columns: Library size, UMI/Library size, MT-gene proportion, batch id
        # assume that library size acts on marginal effect via directly scaling?
        # also: uniform zero-inflation parameter seems not sensible: it does vary across genes and samples
        super(GPerturb_Gaussian, self).__init__()
        self.output_dim = output_dim
        self.hidden_node = hidden_node
        self.base_dim = base_dim
        self.data_size = data_size
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.conditioner_dim = conditioner_dim
        net = [nn.Linear(self.conditioner_dim, self.hidden_node)]
        for _ in range(self.hidden_layer_1):
            net += [nn.ReLU(), nn.Linear(self.hidden_node, self.hidden_node)]
        net += [nn.Linear(self.hidden_node, 6*self.output_dim)]
        self.net = nn.Sequential(*net)

        if base_dim == 0:
            self.net_base = None
        else:
            net_base = [nn.Linear(self.base_dim, self.hidden_node)]
            for _ in range(self.hidden_layer_2):
                net_base += [nn.ReLU(), nn.Linear(self.hidden_node, self.hidden_node)]
            net_base += [nn.Linear(self.hidden_node, self.output_dim)]
            self.net_base = nn.Sequential(*net_base)
        self.net_base_const = nn.Parameter(torch.zeros(self.output_dim))
        self.base_log_var = nn.Parameter(torch.zeros(self.output_dim))
        self.tau = tau
        self.test_id = None

    def forward(self, conditioner, cell_info=None):
        cond_norm = torch.linalg.norm(conditioner, ord=2, dim=1, keepdim=True)
        cond_output = self.net(conditioner).reshape(conditioner.shape[0], 6, self.output_dim)
        mu_mean, mu_log_var, gamma_mean, gamma_log_var, logit_p, logit_p_log_var = \
            cond_norm*cond_output[:, 0, :], cond_norm*cond_output[:, 1, :], cond_norm*cond_output[:, 2, :], \
                cond_norm*cond_output[:, 3, :], cond_output[:, 4, :], cond_output[:, 5, :]
        if cell_info is not None:
            base_mean = self.net_base(cell_info) + self.net_base_const
        else:
            base_mean = self.net_base_const

        return mu_mean, mu_log_var, gamma_mean, gamma_log_var, logit_p, logit_p_log_var, base_mean

    def normal_loss(self, observation, conditioner, cell_info, nu_1, nu_2, nu_3, nu_4, nu_5, nu_6,
                    KL_factor=1., test=False, extra_reg=0.01):
        device = observation.device
        z_gp_mean = torch.tensor(-1.5).to(device)
        if cell_info is not None:
            base_mean_func_val = self.net_base(cell_info)
            base_mean = base_mean_func_val + self.net_base_const
        else:
            base_mean_func_val = torch.tensor(0.).to(observation.device)
            base_mean = self.net_base_const
        base_var_para = self.base_log_var

        cond_norm = torch.linalg.norm(conditioner, ord=2, dim=1, keepdim=True)
        cond_output = self.net(conditioner).reshape(conditioner.shape[0], 6, self.output_dim)
        mu_mean, mu_log_var, gamma_mean, gamma_log_var, logit_p_mean, logit_p_log_var = \
            cond_norm * cond_output[:, 0, :], cond_norm * cond_output[:, 1, :], cond_norm * cond_output[:, 2, :], \
            cond_norm * cond_output[:, 3, :], cond_output[:, 4, :], cond_output[:, 5, :]

        logit_p = logit_p_mean + torch.exp(0.5 * logit_p_log_var) * torch.randn_like(logit_p_mean)
        soft_binary = F.gumbel_softmax(torch.stack((logit_p, torch.zeros_like(logit_p)), dim=-1), hard=False,
                                       tau=self.tau)[:, :, 0]  # batch_size * output_dim

        aux_cond_mean_para = mu_mean + torch.exp(0.5 * mu_log_var) * torch.randn_like(mu_log_var)
        aux_cond_var_para = gamma_mean + torch.exp(0.5 * gamma_log_var) * torch.randn_like(gamma_log_var)

        cond_mean = base_mean + soft_binary * aux_cond_mean_para
        cond_var_para = base_var_para + soft_binary * aux_cond_var_para
        cond_var = logexpp1(cond_var_para)
        cond_log_var = loglogexpp1(cond_var_para)

        unique, idx, counts = torch.unique(conditioner, dim=0, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]).to(device), cum_sum[:-1]))
        unique_id = ind_sorted[cum_sum]
        unique_cond = conditioner[unique_id]

        unique_cond_norm = torch.sum(unique_cond ** 2, dim=1)
        gram = (unique_cond_norm[:, None] + unique_cond_norm[None, :] - 2 * torch.matmul(unique_cond, unique_cond.T))
        sig_1_inv, sig_2_inv, sig_3_inv = torch.linalg.inv(nu_1 * torch.exp(-gram / nu_2)), \
            torch.linalg.inv(nu_3 * torch.exp(-gram / nu_4)), torch.linalg.inv(nu_5 * torch.exp(-gram / nu_6))

        kl_1 = 0.5 * (((sig_1_inv @ mu_mean[unique_id]) * mu_mean[unique_id]).sum() +
                      (torch.exp(mu_log_var[unique_id]) * torch.diag(sig_1_inv).unsqueeze(-1)).sum() -
                      (mu_log_var[unique_id]).sum()) / (cond_mean.shape[1])
        kl_2 = 0.5 * (((sig_2_inv @ gamma_mean[unique_id]) * gamma_mean[unique_id]).sum() +
                      (torch.exp(gamma_log_var[unique_id]) * torch.diag(sig_2_inv).unsqueeze(-1)).sum() -
                      (gamma_log_var[unique_id]).sum()) / (cond_mean.shape[1])
        kl_3 = 0.5 * (((sig_3_inv @ (logit_p_mean[unique_id] - z_gp_mean)) * (logit_p_mean[unique_id] - z_gp_mean)).sum() +
                      (torch.exp(logit_p_log_var[unique_id]) * torch.diag(sig_3_inv).unsqueeze(-1)).sum() -
                      (logit_p_log_var[unique_id]).sum()) / (cond_mean.shape[1])  # average over proteins,
        # Kl of parameters associated with each column (protein) of the dataset

        kl_full = kl_1 + kl_2 + kl_3
        extra_reg = torch.tensor(extra_reg).to(observation.device)
        normal_log_lkd = -0.5 * ((observation - cond_mean) ** 2 / cond_var).nanmean() - 0.5 * cond_log_var.nanmean()
        # averaged over batch size and protein size, per cell per protein,
        # will rescale to full_size * self.data_size, lkd of full sample vector averaged over protein
        return -1 * (normal_log_lkd * self.data_size) + KL_factor*(kl_full + extra_reg*(base_mean_func_val**2).mean())

    def GPerturb_train(self, epoch, observation, cell_info, perturbation,
                    nu_1=1.0, nu_2=0.1, nu_3=1.0, nu_4=0.1, nu_5=1.0, nu_6=0.1, lr=1e-3, device='cpu'):
        my_dataset = MyDataSet(observation=observation, conditioner=perturbation, cell_info=cell_info)

        testing_idx = set(
            np.random.choice(a=range(observation.shape[0]), size=observation.shape[0] // 8, replace=False))
        training_idx = list(set(range(observation.shape[0])) - testing_idx)
        testing_idx = list(testing_idx)
        training_idx_sampler = torch.utils.data.SubsetRandomSampler(training_idx)
        training_loader = torch.utils.data.DataLoader(my_dataset, batch_size=100, sampler=training_idx_sampler)
        test_observation = observation[testing_idx].to(device)
        test_conditioner = perturbation[testing_idx].to(device)
        test_cell_info = cell_info[testing_idx].to(device)
        self.test_id = testing_idx

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        training_loss = torch.zeros(epoch)
        testing_loss = torch.zeros(epoch)
        for EPOCH in range(epoch):
            curr_training_loss = 0.
            self.train()
            for i, (obs, cond, cell_info) in enumerate(training_loader):
                obs = obs.to(device)
                cond = cond.to(device)
                cell_info = cell_info.to(device)
                loss = self.normal_loss(observation=obs, conditioner=cond, cell_info=cell_info,
                                                    nu_1=nu_1, nu_2=nu_2, nu_3=nu_3, nu_4=nu_4, nu_5=nu_5, nu_6=nu_6)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                curr_training_loss += loss.detach().cpu().item() * obs.shape[0]
            training_loss[EPOCH] = curr_training_loss / len(training_idx)
            self.eval()
            with torch.no_grad():
                testing_loss[EPOCH] = self.normal_loss(observation=test_observation,
                                                                   conditioner=test_conditioner,
                                                                   cell_info=test_cell_info, nu_1=nu_1, nu_2=nu_2,
                                                                   nu_3=nu_3, nu_4=nu_4, nu_5=nu_5, nu_6=nu_6,
                                                                   test=True).detach().cpu().item()
            if EPOCH % 10 == 0:
                print('EPOCH={}, test_error={}'.format(EPOCH, testing_loss[EPOCH]))
                print('EPOCH={}, training_error={}'.format(EPOCH, training_loss[EPOCH]))

class GPerturb_ZIP(nn.Module):
    def __init__(self, conditioner_dim, output_dim, base_dim, data_size, hidden_node, hidden_layer_1, hidden_layer_2, tau):
        # base_dim consists 4 columns: Library size, UMI/Library size, MT-gene proportion, batch id
        # assume that library size acts on marginal effect via directly scaling?
        # also: uniform zero-inflation parameter seems not sensible: it does vary across genes and samples
        super(GPerturb_ZIP, self).__init__()
        self.output_dim = output_dim
        self.hidden_node = hidden_node
        self.base_dim = base_dim
        self.data_size = data_size
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.conditioner_dim = conditioner_dim
        net = [nn.Linear(self.conditioner_dim, self.hidden_node)]
        for _ in range(self.hidden_layer_1):
            net += [nn.ReLU(), nn.Linear(self.hidden_node, self.hidden_node)]
        net += [nn.Linear(self.hidden_node, 4*self.output_dim)]
        self.net = nn.Sequential(*net)

        if base_dim == 0:
            self.net_base = None
        else:
            net_base = [nn.Linear(self.base_dim, self.hidden_node)]
            for _ in range(self.hidden_layer_2):
                net_base += [nn.ReLU(), nn.Linear(self.hidden_node, self.hidden_node)]
            net_base += [nn.Linear(self.hidden_node, self.output_dim)]
            self.net_base = nn.Sequential(*net_base)
        self.net_base_const = nn.Parameter(torch.zeros(self.output_dim))

        self.logit_pi = nn.Parameter(torch.zeros(self.output_dim))  # each feature has its own missing rate
        self.tau = tau
        self.test_id = None

    def forward(self, conditioner, cell_info):
        # self.net(conditioner) has size batch*(3*output), turn it into batch*2*output
        cond_norm = torch.linalg.norm(conditioner, ord=2, dim=1, keepdim=True)
        cond_output = self.net(conditioner).reshape(conditioner.shape[0], 4, self.output_dim)
        mu_mean, mu_log_var, logit_p, logit_p_log_var = \
            cond_norm*cond_output[:, 0, :], cond_norm*cond_output[:, 1, :], cond_output[:, 2, :], cond_output[:, 3, :]
        if cell_info is not None:
            base_mean = self.net_base(cell_info) + self.net_base_const  # baseline rate of a cell
        else:
            base_mean = self.net_base_const

        return mu_mean, mu_log_var, base_mean, logit_p, logit_p_log_var

    def zip_loss(self, observation, conditioner, cell_info, nu_1, nu_2, nu_3, nu_4,
                 KL_factor=1., test=False, extra_reg=0.01):
        device = observation.device
        z_gp_mean = torch.tensor(-1.5).to(device)
        if cell_info is not None:
            base_rate_func_val = self.net_base(cell_info)
            base_rate = base_rate_func_val + self.net_base_const
        else:
            base_rate_func_val = torch.tensor(0.).to(observation.device)
            base_rate = self.net_base_const
        # base_rate = self.net_base(cell_info)  # baseline rate of a cell

        # base_L = cell_info[:, 0:1]  # the library size of it
        cond_norm = torch.linalg.norm(conditioner, ord=2, dim=1, keepdim=True)  # size of the conditioner
        cond_output = self.net(conditioner).reshape(conditioner.shape[0], 4, self.output_dim)  # perturb rate
        mu_mean, mu_log_var, logit_p_mean, logit_p_log_var = \
            cond_norm*cond_output[:, 0, :], cond_norm*cond_output[:, 1, :], cond_output[:, 2, :], cond_output[:, 3, :]
        logit_p = logit_p_mean + torch.exp(0.5 * logit_p_log_var) * torch.randn_like(logit_p_mean)

        soft_binary = F.gumbel_softmax(torch.stack((logit_p, torch.zeros_like(logit_p)), dim=-1), hard=False,
                                       tau=self.tau)[:, :, 0]  # batch_size * output_dim

        aux = mu_mean + torch.exp(0.5*mu_log_var) * torch.randn_like(mu_log_var)
        raw_cond_rate = base_rate + soft_binary * aux
        # batch_size * output_dim
        cond_rate = logexpp1(raw_cond_rate) #  * base_L  # / (self.output_dim * F.sigmoid(self.logit_pi))  # scaled by library size
        log_cond_rate = loglogexpp1(raw_cond_rate) #  + torch.log(base_L)  # - nn.LogSigmoid(self.logit_pi)

        pois_log_lkd = observation*log_cond_rate - cond_rate - torch.lgamma(observation+1.)# batch_size * output_dim Pois lkd

        batch_lkd = pois_log_lkd + nn.LogSigmoid()(self.logit_pi.unsqueeze(0))  # non zero part of mixture
        zero_part_lkd = nn.LogSigmoid()(-1.*self.logit_pi.unsqueeze(0))*torch.ones_like(batch_lkd)  # zero part
        stacked_lkd = torch.logsumexp(torch.stack((batch_lkd, zero_part_lkd), dim=2), dim=2)
        batch_lkd[observation == 0.] = stacked_lkd[observation == 0.]  # only update the 0 entries

        unique, idx, counts = torch.unique(conditioner, dim=0, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]).to(device), cum_sum[:-1]))
        unique_id = ind_sorted[cum_sum]
        unique_cond = conditioner[unique_id]

        unique_cond_norm = torch.sum(unique_cond ** 2, dim=1)
        gram = (unique_cond_norm[:, None] + unique_cond_norm[None, :] - 2 * torch.matmul(unique_cond, unique_cond.T))
        sig_1_inv, sig_2_inv = torch.linalg.inv(nu_1 * torch.exp(-gram / nu_2)), \
            torch.linalg.inv(nu_3 * torch.exp(-gram / nu_4))

        kl_1 = 0.5 * (((sig_1_inv @ mu_mean[unique_id]) * mu_mean[unique_id]).sum() +
                      (torch.exp(mu_log_var[unique_id]) * torch.diag(sig_1_inv).unsqueeze(-1)).sum() -
                      (mu_log_var[unique_id]).sum()) / (cond_rate.shape[1])

        kl_2 = 0.5 * (((sig_2_inv @ (logit_p_mean[unique_id] - z_gp_mean)) * (logit_p_mean[unique_id] - z_gp_mean)).sum() +
                      (torch.exp(logit_p_log_var[unique_id]) * torch.diag(sig_2_inv).unsqueeze(-1)).sum() -
                      (logit_p_log_var[unique_id]).sum()) / (cond_rate.shape[1])  # average over proteins,
        # Kl of parameters associated with each column (protein) of the dataset

        return -1 * (batch_lkd.mean() * self.data_size) + KL_factor*(kl_1 + kl_2 + extra_reg*(base_rate_func_val**2).mean())

    def GPerturb_train(self, epoch, observation, cell_info, perturbation,
                    nu_1=1.0, nu_2=0.1, nu_3=1.0, nu_4=0.1, lr=1e-3, device='cpu'):
        my_dataset = MyDataSet(observation=observation, conditioner=perturbation, cell_info=cell_info)

        testing_idx = set(
            np.random.choice(a=range(observation.shape[0]), size=observation.shape[0] // 8, replace=False))
        training_idx = list(set(range(observation.shape[0])) - testing_idx)
        testing_idx = list(testing_idx)
        training_idx_sampler = torch.utils.data.SubsetRandomSampler(training_idx)
        training_loader = torch.utils.data.DataLoader(my_dataset, batch_size=100, sampler=training_idx_sampler)
        test_observation = observation[testing_idx].to(device)
        test_conditioner = perturbation[testing_idx].to(device)
        test_cell_info = cell_info[testing_idx].to(device)
        self.test_id = testing_idx

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        training_loss = torch.zeros(epoch)
        testing_loss = torch.zeros(epoch)
        for EPOCH in range(epoch):
            curr_training_loss = 0.
            self.train()
            for i, (obs, cond, cell_info) in enumerate(training_loader):
                obs = obs.to(device)
                cond = cond.to(device)
                cell_info = cell_info.to(device)
                loss = self.zip_loss(observation=obs, conditioner=cond, cell_info=cell_info,
                                                    nu_1=nu_1, nu_2=nu_2, nu_3=nu_3, nu_4=nu_4)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                curr_training_loss += loss.detach().cpu().item() * obs.shape[0]
            training_loss[EPOCH] = curr_training_loss / len(training_idx)
            self.eval()
            with torch.no_grad():
                testing_loss[EPOCH] = self.zip_loss(observation=test_observation,
                                                                   conditioner=test_conditioner,
                                                                   cell_info=test_cell_info, nu_1=nu_1, nu_2=nu_2,
                                                                   nu_3=nu_3, nu_4=nu_4,
                                                                   test=True).detach().cpu().item()
            if EPOCH % 10 == 0:
                print('EPOCH={}, test_error={}'.format(EPOCH, testing_loss[EPOCH]))
                print('EPOCH={}, training_error={}'.format(EPOCH, training_loss[EPOCH]))

        return training_loss, testing_loss



def my_plot_ZIP(model, obs, cond, cell_info, full_cond, gene_name, cond_name, x, y, fig1, fig2):
    predicted_mu_mean, _, predicted_base_mean, logit_p, logit_p_log_var = model(cond, cell_info)
    estimated_base_mean = predicted_base_mean.detach().cpu().numpy()  # * zeros[testing_idx].numpy()
    estimated_perturbed_mean = (F.sigmoid(logit_p) * predicted_mu_mean).detach().cpu().numpy()
    # estimated_perturbed_mean = (torch.bernoulli(F.sigmoid(logit_p))*predicted_mu_mean).detach().cpu().numpy()
    estimated_inclusion_prob = F.sigmoid(logit_p).detach().cpu().numpy()
    estimated_perturbed_rate = (logexpp1(torch.tensor(estimated_base_mean + estimated_perturbed_mean))).numpy()
    estimated_base_rate = (logexpp1(torch.tensor(estimated_base_mean))).numpy()

    # how does each condition affect different genes?
    unique_conditions = torch.unique(full_cond, dim=0).to(obs.device)

    cond_lvl_predicted_mu_mean, _, _, logit_p, _ = model(unique_conditions, None)
    estimated_inclusion_prob = F.sigmoid(logit_p).detach().cpu().numpy()
    estimated_inclusion = estimated_inclusion_prob > 0.5

    my_gene_name = np.array(gene_name)[estimated_inclusion.sum(axis=0) > 0]
    estimated_inclusion_prob = estimated_inclusion_prob[:, estimated_inclusion.sum(axis=0) > 0]

    unique_conditions = unique_conditions.detach().cpu().numpy()
    yticks = [0 for _ in range(unique_conditions.shape[0])]
    for i in range(unique_conditions.shape[0]):
        if np.all(unique_conditions[i] == 0):
            yticks[i] = 'Non Targeting'
        else:
            yticks[i] = np.array(cond_name)[unique_conditions[i] == 1][0]
    if fig1 is not None:
        fig, axes = plt.subplot_mosaic(
            [['pert', 'base'], ['z_mean', 'z_hist'], ['pair', 'pair']])  # plt.subplots(nrows=2, ncols=2)

        axes['pert'].scatter(estimated_perturbed_rate[obs.cpu() != 0],
                             obs.cpu()[obs.cpu() != 0].numpy(), alpha=0.1)
        axes['pert'].axline((1, 1), slope=1, c='r')
        axes['pert'].set_xlabel('Entries of the predicted Pois rate given the test set')
        axes['pert'].set_ylabel('Entries of the observed count')
        axes['pert'].set_title('Estimated perturbed mean vs real observation')
        axes['pert'].text(x, y, 'Corr = {}'.format(np.round(np.corrcoef(estimated_perturbed_rate[obs.cpu() != 0],
                                                                        obs.cpu()[obs.cpu() != 0].numpy())[0, 1], 5)))

        axes['base'].scatter(estimated_base_rate[obs.cpu() != 0],
                             obs.cpu()[obs.cpu() != 0].numpy(), alpha=0.1)
        axes['base'].axline((1, 1), slope=1, c='r')
        axes['base'].set_xlabel('Entries of the predicted base poisson rate given the test set')
        axes['base'].set_ylabel('Entries of the true perturbation vector')
        axes['base'].set_title('Estimated base mean vs real observation')
        axes['base'].text(x, y, 'Corr = {}'.format(np.round(np.corrcoef(estimated_base_rate[obs.cpu() != 0],
                                                                        obs.cpu()[obs.cpu() != 0].numpy())[0, 1], 5)))

        axes['z_mean'].plot(np.arange(obs.shape[1]), F.sigmoid(logit_p).mean(dim=0).detach().cpu().numpy())
        axes['z_mean'].set_xlabel('Posterior inclusion mean averaged over samples')
        axes['z_mean'].set_ylabel('Gene id')
        axes['z_mean'].set_title('Gene level posterior inclusion')
        axes['z_mean'].axhline(y=0.5, color='r', linestyle='-')

        axes['z_hist'].hist(F.sigmoid(logit_p).detach().cpu().numpy().ravel())
        axes['z_hist'].set_title("Histogram of all the inclusion probabilities")

        im = axes['pair'].imshow(estimated_inclusion_prob)
        axes['pair'].set_xticks(np.arange(len(my_gene_name)), my_gene_name, rotation=90)
        axes['pair'].set_yticks(np.arange(len(yticks)), yticks)
        axes['pair'].set_title('gene-condition pairs that look interesting')
        fig.colorbar(im, ax=axes['pair'])
        fig.set_size_inches(16, 16)
        fig.tight_layout()
        plt.savefig(fig1)

    perturb_level = cond_lvl_predicted_mu_mean[:, estimated_inclusion.sum(axis=0) > 0]
    if fig2 is not None:
        fig, axes = plt.subplots(nrows=1, ncols=1)
        axes.boxplot(perturb_level.detach().cpu().numpy())
        axes.set_title('Box plot of the estimated perturbation level of the selected genes')
        axes.set_xticks(np.arange(len(my_gene_name)) + 1, my_gene_name, rotation=90)
        fig.set_size_inches(16, 16)
        fig.tight_layout()
        plt.savefig(fig2)

    return {'non_zero_pred':estimated_perturbed_rate[obs.cpu() != 0], 'non_zero_obs': obs.cpu()[obs.cpu() != 0].numpy()}


def my_plot_Gaussian(model, obs, cond, cell_info, full_cond, gene_name, cond_name, x, y, fig1, fig2, p=0.9):
    predicted_mu_mean, predicted_mu_var, predicted_gamma_mean, predicted_gamma_var, \
        logit_p, logit_p_log_var, predicted_base_mean = model(cond, cell_info)
    estimated_base_mean = predicted_base_mean.detach().cpu().numpy()  # * zeros[testing_idx].numpy()
    sparse_pert = ((1.0 * (F.sigmoid(logit_p) > p)) * predicted_mu_mean).detach().cpu().numpy()
    estimated_perturbed_mean = ((1.0 * (F.sigmoid(logit_p) > p)) * predicted_mu_mean).detach().cpu().numpy()
    # estimated_perturbed_mean = (torch.bernoulli(F.sigmoid(logit_p))*predicted_mu_mean).detach().cpu().numpy()
    estimated_inclusion_prob = F.sigmoid(logit_p).detach().cpu().numpy()
    estimated_perturbed_mean = (estimated_perturbed_mean + estimated_base_mean)
    estimated_base_var = logexpp1(model.base_log_var).detach().cpu().numpy()
    estimated_perturbed_var = logexpp1(model.base_log_var + (
                1.0 * (F.sigmoid(logit_p) > 0.5)) * predicted_gamma_mean).detach().cpu().numpy()

    # how does each condition affect different genes?
    unique_conditions = torch.unique(full_cond.to(logit_p.device), dim=0)
    perturb_level, _, _, _, logit_p, _, _ = model(unique_conditions, None)
    estimated_inclusion_prob = F.sigmoid(logit_p).detach().cpu().numpy()
    estimated_inclusion = estimated_inclusion_prob > p
    my_gene_name = np.array(gene_name)[estimated_inclusion.sum(axis=0) > 0]
    estimated_inclusion_prob = estimated_inclusion_prob[:, estimated_inclusion.mean(axis=0) > 0]

    unique_conditions = unique_conditions.cpu().numpy()
    my_yticks = ['' for _ in range(unique_conditions.shape[0])]
    for i in range(unique_conditions.shape[0]):
        if np.all(unique_conditions[i] == 0):
            my_yticks[i] = 'Non Targeting'
        else:
            my_yticks[i] = np.array(cond_name)[unique_conditions[i] == 1][0]

    if fig1 is not None:
        fig, axes = plt.subplot_mosaic(
            [['pert', 'base'], ['z_mean', 'z_hist'], ['pair', 'pair']])  # plt.subplots(nrows=2, ncols=2)
        obs = obs.cpu()
        axes['pert'].scatter(estimated_perturbed_mean.ravel(), obs.numpy().ravel(), alpha=0.1)
        axes['pert'].axline((1, 1), slope=1, c='r')
        axes['pert'].set_xlabel('Entries of the predicted perturbed deviance given the test set')
        axes['pert'].set_ylabel('Entries of the observed deviance')
        axes['pert'].set_title('Estimated perturbed mean vs real observation on test set')
        axes['pert'].text(x, y, 'Corr = {}'.format(np.round(np.corrcoef(estimated_perturbed_mean.ravel(),
                                                                              obs.numpy().ravel())[0, 1], 5)))

        axes['base'].scatter(estimated_base_mean.ravel(), obs.numpy().ravel(), alpha=0.1)
        axes['base'].axline((1, 1), slope=1, c='r')
        axes['base'].set_xlabel('Entries of the predicted base level deviance given the test set')
        axes['base'].set_ylabel('Entries of the observed deviance')
        axes['base'].set_title('Estimated base level deviance vs real observation')
        axes['base'].text(x, y, 'Corr = {}'.format(np.round(np.corrcoef(estimated_base_mean.ravel(),
                                                                              obs.numpy().ravel())[0, 1], 5)))

        axes['z_mean'].plot(np.arange(obs.shape[1]), F.sigmoid(logit_p).mean(dim=0).detach().cpu().numpy())
        axes['z_mean'].set_xlabel('Posterior inclusion probability for each gene averaged over all samples')
        axes['z_mean'].set_ylabel('Gene id')
        axes['z_mean'].set_title('Gene level posterior inclusion')
        axes['z_mean'].axhline(y=0.5, color='r', linestyle='-')

        axes['z_hist'].hist(F.sigmoid(logit_p).detach().cpu().numpy().ravel())
        axes['z_hist'].set_title("Histogram of all the inclusion probabilities")

        im = axes['pair'].imshow(estimated_inclusion_prob)
        axes['pair'].axline((4, 4), slope=0, c='r', alpha=0.3)
        axes['pair'].set_xticks(np.arange(len(my_gene_name)), my_gene_name, rotation=90)
        axes['pair'].set_yticks(np.arange(len(my_yticks)), my_yticks)
        axes['pair'].set_title('gene-condition pairs that look interesting')
        fig.colorbar(im, ax=axes['pair'])
        fig.set_size_inches(16, 16)
        fig.tight_layout()
        plt.savefig(fig1)
        plt.close()

    if fig2 is not None:
        fig, axes = plt.subplots(nrows=1, ncols=1)
        axes.boxplot(perturb_level[:, estimated_inclusion.sum(axis=0) > 0].detach().cpu().numpy())
        axes.set_title('Perturbation strength of all unique conditions on the interesting genes expression dims')
        axes.set_xticks(np.arange(len(my_gene_name)) + 1, my_gene_name, rotation=90)
        fig.set_size_inches(16, 16)
        fig.tight_layout()
        plt.savefig(fig2)
        plt.close()

    return {'pert_mean': estimated_perturbed_mean, 'obs': obs.cpu().numpy(), 'prob': F.sigmoid(logit_p).detach().cpu().numpy(),
            'base_removed': obs.cpu().numpy() - estimated_base_mean, 'sparse_pert': sparse_pert,
            'pert_var': estimated_perturbed_var}


def Gaussian_estimates(model, obs, cond, cell_info):
    predicted_mu_mean, predicted_mu_var, predicted_gamma_mean, predicted_gamma_var, \
        logit_p, logit_p_log_var, predicted_base_mean = model(cond, cell_info)
    estimated_base_mean = predicted_base_mean.detach().cpu().numpy()  # * zeros[testing_idx].numpy()
    estimated_perturbed_mean = (F.sigmoid(logit_p) * predicted_mu_mean).detach().cpu().numpy()
    # estimated_perturbed_mean = (torch.bernoulli(F.sigmoid(logit_p))*predicted_mu_mean).detach().cpu().numpy()
    estimated_inclusion_prob = F.sigmoid(logit_p).detach().cpu().numpy()
    estimated_total_mean = (estimated_perturbed_mean + estimated_base_mean)
    estimated_perturbed_var = logexpp1(model.base_log_var + F.sigmoid(logit_p) * predicted_gamma_mean).detach().cpu().numpy()

    return {'pert_mean': estimated_total_mean, 'obs': obs.cpu().numpy(), 'prob': F.sigmoid(logit_p).detach().cpu().numpy(),
            'base_removed': obs.cpu().numpy() - estimated_base_mean, 'pert_effect': estimated_perturbed_mean,
            'pert_var': estimated_perturbed_var}



# alais of GPerturb_Gaussian
class GPerturb_gaussian(nn.Module):
    def __init__(self, conditioner_dim, output_dim, base_dim, data_size, hidden_node, hidden_layer_1, hidden_layer_2, tau):
        # base_dim consists 4 columns: Library size, UMI/Library size, MT-gene proportion, batch id
        # assume that library size acts on marginal effect via directly scaling?
        # also: uniform zero-inflation parameter seems not sensible: it does vary across genes and samples
        super(GPerturb_gaussian, self).__init__()
        self.output_dim = output_dim
        self.hidden_node = hidden_node
        self.base_dim = base_dim
        self.data_size = data_size
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.conditioner_dim = conditioner_dim
        net = [nn.Linear(self.conditioner_dim, self.hidden_node)]
        for _ in range(self.hidden_layer_1):
            net += [nn.ReLU(), nn.Linear(self.hidden_node, self.hidden_node)]
        net += [nn.Linear(self.hidden_node, 6*self.output_dim)]
        self.net = nn.Sequential(*net)

        if base_dim == 0:
            self.net_base = None
            self.net_base_const = nn.Parameter(torch.zeros(self.output_dim))
        else:
            net_base = [nn.Linear(self.base_dim, self.hidden_node)]
            for _ in range(self.hidden_layer_2):
                net_base += [nn.ReLU(), nn.Linear(self.hidden_node, self.hidden_node)]
            net_base += [nn.Linear(self.hidden_node, self.output_dim)]
            self.net_base = nn.Sequential(*net_base)
            self.net_base_const = 0.
        self.base_log_var = nn.Parameter(torch.zeros(self.output_dim))
        self.tau = tau
        self.test_id = None

    def forward(self, conditioner, cell_info=None):
        # self.net(conditioner) has size batch*(3*output), turn it into batch*2*output
        cond_norm = torch.linalg.norm(conditioner, ord=2, dim=1, keepdim=True)
        cond_output = self.net(conditioner).reshape(conditioner.shape[0], 6, self.output_dim)
        mu_mean, mu_log_var, gamma_mean, gamma_log_var, logit_p, logit_p_log_var = \
            cond_norm*cond_output[:, 0, :], cond_norm*cond_output[:, 1, :], cond_norm*cond_output[:, 2, :], \
                cond_norm*cond_output[:, 3, :], cond_output[:, 4, :], cond_output[:, 5, :]
        if cell_info is not None:
            base_mean = self.net_base(cell_info) + self.net_base_const
        else:
            base_mean = self.net_base_const

        return mu_mean, mu_log_var, gamma_mean, gamma_log_var, logit_p, logit_p_log_var, base_mean

    def normal_loss(self, observation, conditioner, cell_info, nu_1, nu_2, nu_3, nu_4, nu_5, nu_6,
                    KL_factor=1., test=False, extra_reg=0.):
        device = observation.device
        z_gp_mean = torch.tensor(-1.5).to(device)
        if cell_info is not None:
            base_mean_func_val = self.net_base(cell_info)
            base_mean = base_mean_func_val + self.net_base_const
        else:
            base_mean_func_val = torch.tensor(0.).to(observation.device)
            base_mean = self.net_base_const
        base_var_para = self.base_log_var

        cond_norm = torch.linalg.norm(conditioner, ord=2, dim=1, keepdim=True)
        cond_output = self.net(conditioner).reshape(conditioner.shape[0], 6, self.output_dim)
        mu_mean, mu_log_var, gamma_mean, gamma_log_var, logit_p_mean, logit_p_log_var = \
            cond_norm * cond_output[:, 0, :], cond_norm * cond_output[:, 1, :], cond_norm * cond_output[:, 2, :], \
            cond_norm * cond_output[:, 3, :], cond_output[:, 4, :], cond_output[:, 5, :]

        logit_p = logit_p_mean + torch.exp(0.5 * logit_p_log_var) * torch.randn_like(logit_p_mean)
        soft_binary = F.gumbel_softmax(torch.stack((logit_p, torch.zeros_like(logit_p)), dim=-1), hard=False,
                                       tau=self.tau)[:, :, 0]  # batch_size * output_dim

        aux_cond_mean_para = mu_mean + torch.exp(0.5 * mu_log_var) * torch.randn_like(mu_log_var)
        aux_cond_var_para = gamma_mean + torch.exp(0.5 * gamma_log_var) * torch.randn_like(gamma_log_var)

        cond_mean = base_mean + soft_binary * aux_cond_mean_para
        cond_var_para = base_var_para + soft_binary * aux_cond_var_para
        cond_var = logexpp1(cond_var_para)
        cond_log_var = loglogexpp1(cond_var_para)

        unique, idx, counts = torch.unique(conditioner, dim=0, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]).to(device), cum_sum[:-1]))
        unique_id = ind_sorted[cum_sum]
        unique_cond = conditioner[unique_id]

        unique_cond_norm = torch.sum(unique_cond ** 2, dim=1)
        gram = (unique_cond_norm[:, None] + unique_cond_norm[None, :] - 2 * torch.matmul(unique_cond, unique_cond.T))
        sig_1_inv, sig_2_inv, sig_3_inv = torch.linalg.inv(nu_1 * torch.exp(-gram / nu_2)), \
            torch.linalg.inv(nu_3 * torch.exp(-gram / nu_4)), torch.linalg.inv(nu_5 * torch.exp(-gram / nu_6))

        kl_1 = 0.5 * (((sig_1_inv @ mu_mean[unique_id]) * mu_mean[unique_id]).sum() +
                      (torch.exp(mu_log_var[unique_id]) * torch.diag(sig_1_inv).unsqueeze(-1)).sum() -
                      (mu_log_var[unique_id]).sum()) / (cond_mean.shape[1])
        kl_2 = 0.5 * (((sig_2_inv @ gamma_mean[unique_id]) * gamma_mean[unique_id]).sum() +
                      (torch.exp(gamma_log_var[unique_id]) * torch.diag(sig_2_inv).unsqueeze(-1)).sum() -
                      (gamma_log_var[unique_id]).sum()) / (cond_mean.shape[1])
        kl_3 = 0.5 * (((sig_3_inv @ (logit_p_mean[unique_id] - z_gp_mean)) * (logit_p_mean[unique_id] - z_gp_mean)).sum() +
                      (torch.exp(logit_p_log_var[unique_id]) * torch.diag(sig_3_inv).unsqueeze(-1)).sum() -
                      (logit_p_log_var[unique_id]).sum()) / (cond_mean.shape[1])  # average over proteins,
        # Kl of parameters associated with each column (protein) of the dataset

        kl_full = kl_1 + kl_2 + kl_3
        extra_reg = torch.tensor(extra_reg).to(observation.device)
        normal_log_lkd = -0.5 * ((observation - cond_mean) ** 2 / cond_var).mean() - 0.5 * cond_log_var.mean()
        # averaged over batch size and protein size, per cell per protein,
        # will rescale to full_size * self.data_size, lkd of full sample vector averaged over protein

        if test:
            print(normal_log_lkd * self.data_size, kl_1, kl_2, kl_3)

        return -1 * (normal_log_lkd * self.data_size) + KL_factor*(kl_full + extra_reg*(base_mean_func_val**2).mean())

    def GPerturb_train(self, epoch, observation, cell_info, perturbation,
                       nu_1=1.0, nu_2=0.1, nu_3=1.0, nu_4=0.1, nu_5=1.0, nu_6=0.1, lr=1e-3, device='cpu'):
        my_dataset = MyDataSet(observation=observation, conditioner=perturbation, cell_info=cell_info)

        testing_idx = set(
            np.random.choice(a=range(observation.shape[0]), size=observation.shape[0] // 8, replace=False))
        training_idx = list(set(range(observation.shape[0])) - testing_idx)
        testing_idx = list(testing_idx)
        training_idx_sampler = torch.utils.data.SubsetRandomSampler(training_idx)
        training_loader = torch.utils.data.DataLoader(my_dataset, batch_size=100, sampler=training_idx_sampler)
        test_observation = observation[testing_idx].to(device)
        test_conditioner = perturbation[testing_idx].to(device)
        test_cell_info = cell_info[testing_idx].to(device)
        self.test_id = testing_idx

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        training_loss = torch.zeros(epoch)
        testing_loss = torch.zeros(epoch)
        for EPOCH in range(epoch):
            curr_training_loss = 0.
            self.train()
            for i, (obs, cond, cell_info) in enumerate(training_loader):
                obs = obs.to(device)
                cond = cond.to(device)
                cell_info = cell_info.to(device)
                loss = self.normal_loss(observation=obs, conditioner=cond, cell_info=cell_info,
                                                    nu_1=nu_1, nu_2=nu_2, nu_3=nu_3, nu_4=nu_4, nu_5=nu_5, nu_6=nu_6)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                curr_training_loss += loss.detach().cpu().item() * obs.shape[0]
            training_loss[EPOCH] = curr_training_loss / len(training_idx)
            self.eval()
            with torch.no_grad():
                testing_loss[EPOCH] = self.normal_loss(observation=test_observation,
                                                                   conditioner=test_conditioner,
                                                                   cell_info=test_cell_info, nu_1=nu_1, nu_2=nu_2,
                                                                   nu_3=nu_3, nu_4=nu_4, nu_5=nu_5, nu_6=nu_6,
                                                                   test=True).detach().cpu().item()
            if EPOCH % 10 == 0:
                print('EPOCH={}, test_error={}'.format(EPOCH, testing_loss[EPOCH]))
                print('EPOCH={}, training_error={}'.format(EPOCH, training_loss[EPOCH]))






class GPerturb_ZINB(nn.Module):
    def __init__(self, conditioner_dim, output_dim, base_dim, data_size, hidden_node, hidden_layer_1, hidden_layer_2, tau):
        # base_dim consists 4 columns: Library size, UMI/Library size, MT-gene proportion, batch id
        # assume that library size acts on marginal effect via directly scaling?
        # also: uniform zero-inflation parameter seems not sensible: it does vary across genes and samples
        super(GPerturb_ZINB, self).__init__()
        self.output_dim = output_dim
        self.hidden_node = hidden_node
        self.base_dim = base_dim
        self.data_size = data_size
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.conditioner_dim = conditioner_dim
        net = [nn.Linear(self.conditioner_dim, self.hidden_node)]
        for _ in range(self.hidden_layer_1):
            net += [nn.ReLU(), nn.Linear(self.hidden_node, self.hidden_node)]
        net += [nn.Linear(self.hidden_node, 4*self.output_dim)]
        self.net = nn.Sequential(*net)

        if base_dim == 0:
            self.net_base = None
        else:
            net_base = [nn.Linear(self.base_dim, self.hidden_node)]
            for _ in range(self.hidden_layer_2):
                net_base += [nn.ReLU(), nn.Linear(self.hidden_node, self.hidden_node)]
            net_base += [nn.Linear(self.hidden_node, self.output_dim)]
            self.net_base = nn.Sequential(*net_base)
        self.net_base_const = nn.Parameter(torch.zeros(self.output_dim))

        self.log_dispersion = nn.Parameter(torch.zeros(self.output_dim))  # each feature has its own dispersion
        self.logit_pi = nn.Parameter(torch.zeros(self.output_dim))  # each feature has its own missing rate
        self.tau = tau
        self.test_id = None

    def forward(self, conditioner, cell_info):
        # self.net(conditioner) has size batch*(3*output), turn it into batch*2*output
        cond_norm = torch.linalg.norm(conditioner, ord=2, dim=1, keepdim=True)
        cond_output = self.net(conditioner).reshape(conditioner.shape[0], 4, self.output_dim)
        mu_mean, mu_log_var, logit_p, logit_p_log_var = \
            cond_norm*cond_output[:, 0, :], cond_norm*cond_output[:, 1, :], cond_output[:, 2, :], cond_output[:, 3, :]
        if cell_info is not None:
            base_mean = self.net_base(cell_info) + self.net_base_const  # baseline rate of a cell
        else:
            base_mean = self.net_base_const

        return mu_mean, mu_log_var, base_mean, logit_p, logit_p_log_var

    def zinb_loss(self, observation, conditioner, cell_info, nu_1, nu_2, nu_3, nu_4,
                 KL_factor=1., test=False, extra_reg=0.01):
        device = observation.device
        z_gp_mean = torch.tensor(-1.5).to(device)
        if cell_info is not None:
            base_rate_func_val = self.net_base(cell_info)
            base_rate = base_rate_func_val + self.net_base_const
        else:
            base_rate_func_val = torch.tensor(0.).to(observation.device)
            base_rate = self.net_base_const
        # base_rate = self.net_base(cell_info)  # baseline rate of a cell

        # base_L = cell_info[:, 0:1]  # the library size of it
        cond_norm = torch.linalg.norm(conditioner, ord=2, dim=1, keepdim=True)  # size of the conditioner
        cond_output = self.net(conditioner).reshape(conditioner.shape[0], 4, self.output_dim)  # perturb rate
        mu_mean, mu_log_var, logit_p_mean, logit_p_log_var = \
            cond_norm*cond_output[:, 0, :], cond_norm*cond_output[:, 1, :], cond_output[:, 2, :], cond_output[:, 3, :]
        logit_p = logit_p_mean + torch.exp(0.5 * logit_p_log_var) * torch.randn_like(logit_p_mean)

        soft_binary = F.gumbel_softmax(torch.stack((logit_p, torch.zeros_like(logit_p)), dim=-1), hard=False,
                                       tau=self.tau)[:, :, 0]  # batch_size * output_dim

        aux = mu_mean + torch.exp(0.5*mu_log_var) * torch.randn_like(mu_log_var)
        raw_cond_rate = base_rate + soft_binary * aux
        # batch_size * output_dim
        cond_rate = logexpp1(raw_cond_rate) #  * base_L  # / (self.output_dim * F.sigmoid(self.logit_pi))  # scaled by library size
        log_cond_rate = loglogexpp1(raw_cond_rate) #  + torch.log(base_L)  # - nn.LogSigmoid(self.logit_pi)

        dispersion = torch.exp(self.log_dispersion)
        gamma_pois_lkd = observation * (self.log_dispersion + log_cond_rate) - \
                         (observation + 1 / dispersion) * torch.log(1 + cond_rate * dispersion) - \
                         torch.lgamma(observation + 1.) - torch.lgamma(1 / dispersion) + \
                         torch.lgamma(observation + 1 / dispersion)

        # pois_log_lkd = observation*log_cond_rate - cond_rate - torch.lgamma(observation+1.)# batch_size * output_dim Pois lkd

        batch_lkd = gamma_pois_lkd + nn.LogSigmoid()(self.logit_pi.unsqueeze(0))  # non zero part of mixture
        zero_part_lkd = nn.LogSigmoid()(-1.*self.logit_pi.unsqueeze(0))*torch.ones_like(batch_lkd)  # zero part
        stacked_lkd = torch.logsumexp(torch.stack((batch_lkd, zero_part_lkd), dim=2), dim=2)
        batch_lkd[observation == 0.] = stacked_lkd[observation == 0.]  # only update the 0 entries

        unique, idx, counts = torch.unique(conditioner, dim=0, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]).to(device), cum_sum[:-1]))
        unique_id = ind_sorted[cum_sum]
        unique_cond = conditioner[unique_id]

        unique_cond_norm = torch.sum(unique_cond ** 2, dim=1)
        gram = (unique_cond_norm[:, None] + unique_cond_norm[None, :] - 2 * torch.matmul(unique_cond, unique_cond.T))
        sig_1_inv, sig_2_inv = torch.linalg.inv(nu_1 * torch.exp(-gram / nu_2)), \
            torch.linalg.inv(nu_3 * torch.exp(-gram / nu_4))

        kl_1 = 0.5 * (((sig_1_inv @ mu_mean[unique_id]) * mu_mean[unique_id]).sum() +
                      (torch.exp(mu_log_var[unique_id]) * torch.diag(sig_1_inv).unsqueeze(-1)).sum() -
                      (mu_log_var[unique_id]).sum()) / (cond_rate.shape[1])

        kl_2 = 0.5 * (((sig_2_inv @ (logit_p_mean[unique_id] - z_gp_mean)) * (logit_p_mean[unique_id] - z_gp_mean)).sum() +
                      (torch.exp(logit_p_log_var[unique_id]) * torch.diag(sig_2_inv).unsqueeze(-1)).sum() -
                      (logit_p_log_var[unique_id]).sum()) / (cond_rate.shape[1])  # average over proteins,
        # Kl of parameters associated with each column (protein) of the dataset

        return -1 * (batch_lkd.mean() * self.data_size) + KL_factor*(kl_1 + kl_2 + extra_reg*(base_rate_func_val**2).mean())

    def GPerturb_train(self, epoch, observation, cell_info, perturbation,
                    nu_1=1.0, nu_2=0.1, nu_3=1.0, nu_4=0.1, lr=1e-3, device='cpu'):
        my_dataset = MyDataSet(observation=observation, conditioner=perturbation, cell_info=cell_info)

        testing_idx = set(
            np.random.choice(a=range(observation.shape[0]), size=observation.shape[0] // 8, replace=False))
        training_idx = list(set(range(observation.shape[0])) - testing_idx)
        testing_idx = list(testing_idx)
        training_idx_sampler = torch.utils.data.SubsetRandomSampler(training_idx)
        training_loader = torch.utils.data.DataLoader(my_dataset, batch_size=100, sampler=training_idx_sampler)
        test_observation = observation[testing_idx].to(device)
        test_conditioner = perturbation[testing_idx].to(device)
        test_cell_info = cell_info[testing_idx].to(device)
        self.test_id = testing_idx

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        training_loss = torch.zeros(epoch)
        testing_loss = torch.zeros(epoch)
        for EPOCH in range(epoch):
            curr_training_loss = 0.
            self.train()
            for i, (obs, cond, cell_info) in enumerate(training_loader):
                obs = obs.to(device)
                cond = cond.to(device)
                cell_info = cell_info.to(device)
                loss = self.zinb_loss(observation=obs, conditioner=cond, cell_info=cell_info,
                                                    nu_1=nu_1, nu_2=nu_2, nu_3=nu_3, nu_4=nu_4)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                curr_training_loss += loss.detach().cpu().item() * obs.shape[0]
            training_loss[EPOCH] = curr_training_loss / len(training_idx)
            self.eval()
            with torch.no_grad():
                testing_loss[EPOCH] = self.zip_loss(observation=test_observation,
                                                                   conditioner=test_conditioner,
                                                                   cell_info=test_cell_info, nu_1=nu_1, nu_2=nu_2,
                                                                   nu_3=nu_3, nu_4=nu_4,
                                                                   test=True).detach().cpu().item()
            if EPOCH % 10 == 0:
                print('EPOCH={}, test_error={}'.format(EPOCH, testing_loss[EPOCH]))
                print('EPOCH={}, training_error={}'.format(EPOCH, training_loss[EPOCH]))

        return training_loss, testing_loss
