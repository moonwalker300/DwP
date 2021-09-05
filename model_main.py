# 这里只有IVAE和CEVAE有用，其他的XXXVAE都忽略吧
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F


def weights_init(m):
    if isinstance(m, nn.Linear):
        #nn.init.xavier_uniform_(m.weight.data, gain=0.3)
        nn.init.constant_(m.bias.data, 0)


def BinaryCE(pre, tar):
    return -(tar * torch.log(pre + 1e-4) + (1 - tar) * torch.log(1 - pre + 1e-4))


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, sig=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(
                nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim)
            )
        self.fc = nn.ModuleList(_fc_list)
        self.sig = sig
        self.apply(weights_init)

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = F.elu(self.fc[c](h))
        if self.sig:
            h = torch.sigmoid(h)
        return h


class CVAE:
    def __init__(
        self,
        obsx_dim,
        latent_dim,
        treat_dim,
        hidden_dim,
        n_layers,
        y_layers,
        y_hidden,
        learning_rate=0.001,
        y_cof=2.0,
    ):
        self.hid_prior_mean = MLP(obsx_dim, latent_dim, hidden_dim, n_layers)
        self.hid_prior_logv = MLP(obsx_dim, latent_dim, hidden_dim, n_layers)
        self.encoder_mean = MLP(
            obsx_dim + treat_dim + 1, latent_dim, hidden_dim, n_layers
        )
        self.encoder_logv = MLP(
            obsx_dim + treat_dim + 1, latent_dim, hidden_dim, n_layers
        )
        self.decoder_t = MLP(
            obsx_dim + latent_dim, treat_dim, hidden_dim, n_layers, True
        )
        self.decoder_y = MLP(obsx_dim + latent_dim + treat_dim, 1, y_hidden, y_layers)
        self.decoder_y_logv = MLP(
            obsx_dim + latent_dim + treat_dim, 1, y_hidden, y_layers
        )
        models = [
            self.hid_prior_mean,
            self.hid_prior_logv,
            self.encoder_mean,
            self.encoder_logv,
            self.decoder_t,
            self.decoder_y,
            self.decoder_y_logv,
        ]
        self.bceloss = nn.BCELoss(reduction="none")
        self.mseloss = nn.MSELoss(reduction="none")
        parameters = []
        for model in models:
            parameters.extend(list(model.parameters()))
        self.optimizer = optim.Adam(parameters, lr=learning_rate)
        self.decoder_test = MLP(
            obsx_dim + latent_dim + treat_dim, 1, y_hidden, y_layers
        )
        self.optimizer_test = optim.Adam(
            self.decoder_test.parameters(), lr=learning_rate
        )
        self.y_cof = y_cof
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def neg_elbo(self, obsx, t, y):
        prior_m = self.hid_prior_mean(obsx)
        prior_lv = self.hid_prior_logv(obsx)
        z_mean = self.encoder_mean(torch.cat((obsx, t, y), dim=1))
        z_logv = self.encoder_logv(torch.cat((obsx, t, y), dim=1))

        std_z = torch.randn(size=z_mean.size()).to(self.device)
        sample_z = std_z * torch.exp(z_logv) + z_mean

        rec_t = self.decoder_t(torch.cat((obsx, sample_z), dim=1))
        rec_y = self.decoder_y(torch.cat((obsx, sample_z, t), dim=1))
        y_logv = self.decoder_y_logv(torch.cat((obsx, sample_z, t), dim=1))
        t_loss = self.bceloss(rec_t, t).sum(1)
        y_loss = (self.mseloss(rec_y, y) * torch.exp(-2 * y_logv) / 2 + y_logv).sum(1)
        KL_divergence = 0.5 * (
            (prior_lv - z_logv) * 2
            - 1
            + torch.exp(2 * (z_logv - prior_lv))
            + (z_mean - prior_m) * (z_mean - prior_m) * torch.exp(-2 * prior_lv)
        ).sum(1)
        rec_loss = t_loss + y_loss

        return (
            (KL_divergence + rec_loss).mean(),
            KL_divergence.mean(),
            rec_loss.mean(),
            t_loss.mean(),
            y_loss.mean(),
        )

    def reset(self):
        self.decoder_test.apply(weights_init)

    def optimize_test(self, obsx, t, y, y_test):
        z = self.infer_post(obsx, t, y, True)
        y_pre = self.decoder_test(torch.cat((obsx, z, t), dim=1))
        self.optimizer_test.zero_grad()
        loss = (self.mseloss(y_pre, y_test)).mean()
        loss.backward()
        self.optimizer_test.step()
        return loss.item()

    def predict_test(self, obsx, t, y, tnew):
        pre_y = np.zeros(obsx.shape[0])
        for i in range(500):
            z = self.infer_post(obsx, t, y, True)
            tmp = self.decoder_test(torch.cat((obsx, z, tnew), dim=1))
            tmp = tmp.detach().numpy().squeeze()
            pre_y = pre_y + tmp
        pre_y /= 500
        return pre_y

    def optimize(self, obsx, t, y):
        self.optimizer.zero_grad()
        loss, kl, rec, t_loss, y_loss = self.neg_elbo(obsx, t, y)
        loss.backward()
        self.optimizer.step()
        return loss.item(), rec.item(), kl.item(), t_loss.item(), y_loss.item()

    def infer_post(self, obsx, t, y, ifnoise):
        if not ifnoise:
            ret = self.encoder_mean(torch.cat((obsx, t, y), dim=1))
        else:
            ret = self.encoder_mean(torch.cat((obsx, t, y), dim=1))
            ret += torch.exp(
                self.encoder_logv(torch.cat((obsx, t, y), dim=1))
            ) * torch.randn(size=ret.size()).to(self.device)
        return ret

    def predict_post(self, obsx, t, y, tnew, ifexp=True):
        if ifexp:
            z = self.infer_post(obsx, t, y, False)
            pre_y = self.decoder_y(torch.cat((obsx, z, tnew), dim=1))
            pre_y = pre_y.detach().numpy().squeeze()
        else:
            pre_y = np.zeros(obsx.shape[0])
            for i in range(500):
                z = self.infer_post(obsx, t, y, True)
                tmp = self.decoder_y(torch.cat((obsx, z, tnew), dim=1))
                tmp = tmp.detach().numpy().squeeze()
                pre_y = pre_y + tmp
            pre_y /= 500
        return pre_y

    def infer_prior(self, obsx, ifnoise):
        if not ifnoise:
            ret = self.hid_prior_mean(obsx)
        else:
            ret = self.hid_prior_mean(obsx)
            ret += torch.exp(self.hid_prior_logv(obsx)) * torch.randn(size=ret.size()).to(self.device)
        return ret

    def predict_prior(self, obsx, tnew, ifexp=True):
        if ifexp:
            z = self.infer_prior(obsx, False)
            pre_y = self.decoder_y(torch.cat((obsx, z, tnew), dim=1))
            pre_y = pre_y.detach().numpy().squeeze()
        else:
            pre_y = np.zeros(obsx.shape[0])
            for i in range(500):
                z = self.infer_prior(obsx, True)
                tmp = self.decoder_y(torch.cat((obsx, z, tnew), dim=1))
                tmp = tmp.detach().numpy().squeeze()
                pre_y = pre_y + tmp
            pre_y /= 500
        return pre_y


class IVAE(nn.Module):
    def __init__(
        self,
        obsx_dim,
        latent_dim,
        treat_dim,
        hidden_dim,
        n_layers,
        y_layers,
        y_hidden,
        learning_rate=0.001,
        weight_decay=0.001,
        y_cof=2.0,
    ):
        super().__init__()
        self.hid_prior_mean = MLP(obsx_dim, latent_dim, hidden_dim, n_layers)
        self.hid_prior_logv = MLP(obsx_dim, latent_dim, hidden_dim, n_layers)
        self.encoder_mean = MLP(
            obsx_dim + treat_dim + 1, latent_dim, hidden_dim, n_layers
        )
        self.encoder_logv = MLP(
            obsx_dim + treat_dim + 1, latent_dim, hidden_dim, n_layers
        )
        self.decoder_t = MLP(latent_dim, treat_dim, hidden_dim, n_layers, True)
        self.decoder_y = MLP(latent_dim + treat_dim, 1, y_hidden, y_layers)
        models = [
            self.hid_prior_mean,
            self.hid_prior_logv,
            self.encoder_mean,
            self.encoder_logv,
            self.decoder_t,
            self.decoder_y,
        ]
        self.bceloss = nn.BCELoss(reduction="none")
        self.mseloss = nn.MSELoss(reduction="none")
        parameters = []
        for model in models:
            parameters.extend(list(model.parameters()))
        self.optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
        self.y_cof = y_cof
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def neg_elbo(self, obsx, t, y):
        prior_m = self.hid_prior_mean(obsx) * 0 + 1
        prior_lv = self.hid_prior_logv(obsx) * 0 + 1
        z_mean = self.encoder_mean(torch.cat((obsx, t, y), dim=1))
        z_logv = self.encoder_logv(torch.cat((obsx, t, y), dim=1))

        std_z = torch.randn(size=z_mean.size()).to(self.device)
        sample_z = std_z * torch.exp(z_logv) + z_mean

        rec_t = self.decoder_t(sample_z)
        rec_y = self.decoder_y(torch.cat((sample_z, t), dim=1))
        t_loss = self.bceloss(rec_t, t).sum(1)
        y_loss = self.mseloss(rec_y, y).sum(1)
        KL_divergence = 0.5 * (
            (prior_lv - z_logv) * 2
            - 1
            + torch.exp(2 * (z_logv - prior_lv))
            + (z_mean - prior_m) * (z_mean - prior_m) * torch.exp(-2 * prior_lv)
        ).sum(1)
        rec_loss = t_loss + y_loss * self.y_cof

        return (
            (KL_divergence + rec_loss).mean(),
            KL_divergence.mean(),
            rec_loss.mean(),
            t_loss.mean(),
            y_loss.mean(),
        )

    def optimize(self, obsx, t, y):
        self.optimizer.zero_grad()
        loss, kl, rec, t_loss, y_loss = self.neg_elbo(obsx, t, y)
        loss.backward()
        self.optimizer.step()
        return loss.item(), rec.item(), kl.item(), t_loss.item(), y_loss.item()

    def infer_post(self, obsx, t, y, ifnoise):
        if not ifnoise:
            ret = self.encoder_mean(torch.cat((obsx, t, y), dim=1))
        else:
            ret = self.encoder_mean(torch.cat((obsx, t, y), dim=1))
            ret += torch.exp(
                self.encoder_logv(torch.cat((obsx, t, y), dim=1))
            ) * torch.randn(size=ret.size()).to(self.device)
        return ret

    def predict_post(self, obsx, t, y, tnew, ifexp=True):
        if ifexp:
            z = self.infer_post(obsx, t, y, False)
            pre_y = self.decoder_y(torch.cat((z, tnew), dim=1))
            pre_y = pre_y.detach().cpu().numpy().squeeze()
        else:
            pre_y = np.zeros(obsx.shape[0])
            for i in range(500):
                z = self.infer_post(obsx, t, y, True)
                tmp = self.decoder_y(torch.cat((z, tnew), dim=1))
                tmp = tmp.detach().cpu().numpy().squeeze()
                pre_y = pre_y + tmp
            pre_y /= 500
        return pre_y

    def infer_prior(self, obsx, ifnoise):
        if not ifnoise:
            ret = self.hid_prior_mean(obsx)
        else:
            ret = self.hid_prior_mean(obsx)
            ret += torch.exp(self.hid_prior_logv(obsx)) * torch.randn(size=ret.size()).to(self.device)
        return ret, torch.exp(self.hid_prior_logv(obsx))

    def predict_prior(self, obsx, tnew, ifexp=True):
        if ifexp:
            z = self.infer_prior(obsx, False)
            pre_y = self.decoder_y(torch.cat((z, tnew), dim=1))
            pre_y = pre_y.detach().cpu().numpy().squeeze()
        else:
            pre_y = np.zeros(obsx.shape[0])
            for i in range(500):
                z = self.infer_prior(obsx, True)
                tmp = self.decoder_y(torch.cat((z, tnew), dim=1))
                tmp = tmp.detach().cpu().numpy().squeeze()
                pre_y = pre_y + tmp
            pre_y /= 500
        return pre_y


class CEVAE(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_dim,
        output_dim_bin,
        output_dim_con,
        n_layers,
        y_layers,
        y_hidden,
        learning_rate,
        weight_decay=0.001, y_cof = 2.0
    ):
        super().__init__()
        self.decoder_bin = MLP(latent_dim, output_dim_bin, hidden_dim, n_layers, True)
        self.decoder_con = MLP(latent_dim, output_dim_con, hidden_dim, n_layers, False)
        self.decoder_con_logv = MLP(
            latent_dim, output_dim_con, hidden_dim, n_layers, False
        )
        self.decoder_y = MLP(latent_dim + output_dim_bin, 1, y_hidden, y_layers, False)
        # self.decoder_y_logv = MLP(latent_dim + output_dim_bin, 1, y_hidden, y_layers, False)
        self.encoder_mean = MLP(
            output_dim_bin + output_dim_con + 1, latent_dim, hidden_dim, n_layers
        )
        self.encoder_logv = MLP(
            output_dim_bin + output_dim_con + 1, latent_dim, hidden_dim, n_layers
        )
        self.bceloss = nn.BCELoss(reduction="none")
        self.mseloss_x = nn.MSELoss(reduction="none")
        self.mseloss_y = nn.MSELoss(reduction="none")
        parameters = (
            list(self.decoder_bin.parameters())
            + list(self.decoder_con.parameters())
            + list(self.encoder_mean.parameters())
            + list(self.encoder_logv.parameters())
            + list(self.decoder_y.parameters())
            + list(self.decoder_con_logv.parameters())
        )  # + list(self.decoder_y_logv.parameters())
        self.optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.y_cof = y_cof

    def neg_elbo(self, t, obs_x, y):
        t_obsx_y = torch.cat((t, obs_x, y), dim=1)
        z_mean = self.encoder_mean(t_obsx_y)
        z_lv = self.encoder_logv(t_obsx_y)

        std_z = torch.randn(size=z_mean.size()).to(self.device)
        sample_z = std_z * torch.exp(z_lv) + z_mean

        rec_t = self.decoder_bin(sample_z)
        rec_x = self.decoder_con(sample_z)
        rec_x_lv = self.decoder_con_logv(sample_z)
        rec_y = self.decoder_y(torch.cat((sample_z, t), dim=1))
        t_loss = self.bceloss(rec_t, t).sum(1)
        x_loss = (
            self.mseloss_x(rec_x, obs_x) * torch.exp(-2 * rec_x_lv) / 2 + rec_x_lv
        ).sum(1)
        y_loss = (self.mseloss_y(rec_y, y)).sum(1)
        rec_loss = t_loss + y_loss * self.y_cof + x_loss

        KL_divergence = 0.5 * (
            (-z_lv) * 2 - 1 + torch.exp(2 * z_lv) + z_mean * z_mean
        ).sum(1)
        return (
            (rec_loss + KL_divergence).mean(),
            rec_loss.mean(),
            KL_divergence.mean(),
            t_loss.mean(),
            x_loss.mean(),
            y_loss.mean(),
        )

    def optimize(self, t, obs_x, y):
        self.optimizer.zero_grad()
        loss, rec_l, KL_d, t_loss, x_loss, y_loss = self.neg_elbo(t, obs_x, y)
        loss.backward()
        self.optimizer.step()
        return (
            loss.item(),
            rec_l.item(),
            KL_d.item(),
            t_loss.item(),
            x_loss.item(),
            y_loss.item(),
        )

    def predict(self, t, obs_x, y, t_new, ifexp=True):
        if ifexp:
            t_obsx_y = torch.cat((t, obs_x, y), dim=1)
            z_mean = self.encoder_mean(t_obsx_y)
            pre_y = self.decoder_y(torch.cat((z_mean, t_new), dim=1))
            pre_y = pre_y.detach().cpu().numpy().squeeze()
        else:
            pre_y = np.zeros(obs_x.shape[0])
            t_obsx_y = torch.cat((t, obs_x, y), dim=1)
            z_mean = self.encoder_mean(t_obsx_y)
            z_v = torch.exp(self.encoder_logv(t_obsx_y))
            for i in range(500):
                z_sample = z_mean + z_v * torch.randn(size=z_mean.size()).to(self.device)
                tmp = self.decoder_y(torch.cat((z_sample, t_new), dim=1))
                tmp = tmp.detach().cpu().numpy().squeeze()
                pre_y += tmp
            pre_y /= 500
        return pre_y

    def infer(self, t, obs_x, y, ifn=0):
        t_obsx_y = torch.cat((t, obs_x, y), dim=1)
        z_infer = self.encoder_mean(t_obsx_y)
        z_noise = torch.exp(self.encoder_logv(t_obsx_y))
        z_infer += ifn * torch.randn(size=z_infer.size()).to(self.device) * z_noise
        return z_infer


class VAE:
    def __init__(
        self, latent_dim, hidden_dim, output_dim_bin, output_dim_con, n_layers
    ):
        self.decoder_bin = MLP(latent_dim, output_dim_bin, hidden_dim, n_layers, True)
        self.decoder_con = MLP(latent_dim, output_dim_con, hidden_dim, n_layers, False)
        self.encoder_mean = MLP(
            output_dim_bin + output_dim_con, latent_dim, hidden_dim, n_layers
        )
        self.encoder_logv = MLP(
            output_dim_bin + output_dim_con, latent_dim, hidden_dim, n_layers
        )
        self.bceloss = nn.BCELoss(reduction="none")
        self.mseloss = nn.MSELoss(reduction="none")
        parameters = (
            list(self.decoder_bin.parameters())
            + list(self.decoder_con.parameters())
            + list(self.encoder_mean.parameters())
            + list(self.encoder_logv.parameters())
        )
        self.optimizer = optim.Adam(parameters, lr=0.01)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self, z):
        z = torch.FloatTensor(z)
        ret_t = self.decoder_bin(z)
        return torch.bernoulli(ret_t).detach().numpy()

    def propensity_score(self, z, t, sign=1):
        z = torch.FloatTensor(z)
        lh = self.decoder_bin(z).detach().numpy()
        return np.exp(sign * np.sum(t * np.log(lh) + (1 - t) * np.log(1 - lh), axis=1))

    def look(self):
        zz = self.encoder_mean.fc[0].weight
        zz = zz.detach().numpy()
        print(np.sum(np.abs(zz), axis=0))
        zz2 = self.encoder_mean.fc[1].weight
        zz2 = zz2.detach().numpy()
        zz = zz2.dot(zz)
        print(np.sum(np.abs(zz), axis=0))

    def neg_elbo(self, t, obs_x):
        t_obsx = torch.cat((t, obs_x), dim=1)
        z_mean = self.encoder_mean(t_obsx)
        z_lv = self.encoder_logv(t_obsx)

        std_z = torch.randn(size=z_mean.size()).to(self.device)
        sample_z = std_z * torch.exp(z_lv) + z_mean

        rec_t = self.decoder_bin(sample_z)
        rec_x = self.decoder_con(sample_z)

        rec_loss = self.bceloss(rec_t, t).sum(1) + self.mseloss(rec_x, obs_x).sum(1)

        KL_divergence = 0.5 * (
            (-z_lv) * 2 - 1 + torch.exp(2 * z_lv) + z_mean * z_mean
        ).sum(1)
        return (rec_loss + KL_divergence).mean(), rec_loss.mean(), KL_divergence.mean()

    def rec_loss(self, t, obs_x, nl):
        t_obsx = torch.cat((t, obs_x), dim=1)
        z_mean = self.encoder_mean(t_obsx)
        z_lv = self.encoder_logv(t_obsx)

        std_z = torch.randn(size=z_mean.size()).to(self.device)
        sample_z = std_z * torch.exp(z_lv) * nl + z_mean

        rec_t = self.decoder_bin(sample_z)
        rec_loss = self.bceloss(rec_t, t).sum(1)

        return rec_loss.sum()

    def optimize(self, t, obs_x):
        self.optimizer.zero_grad()
        loss, rec_l, KL_d = self.neg_elbo(t, obs_x)
        loss.backward()
        self.optimizer.step()
        return loss.item(), rec_l.item(), KL_d.item()

    def infer(self, t, obs_x, ifn=0):
        t_obsx = torch.cat((t, obs_x), dim=1)
        z_infer = self.encoder_mean(t_obsx)
        z_noise = torch.exp(self.encoder_logv(t_obsx))
        z_infer += ifn * torch.randn(size=z_infer.size()).to(self.device) * z_noise
        # print(torch.cat((self.encoder_mean(t_obsx), torch.exp(self.encoder_logv(t_obsx))), 1).detach().numpy())
        z_infer = z_infer.detach().numpy()
        return z_infer


class TVAE:
    def __init__(self, latent_dim, hidden_dim, output_dim_bin, n_layers):
        self.decoder_bin = MLP(latent_dim, output_dim_bin, hidden_dim, n_layers, True)
        self.encoder_mean = MLP(output_dim_bin, latent_dim, hidden_dim, n_layers)
        self.encoder_logv = MLP(output_dim_bin, latent_dim, hidden_dim, n_layers)
        self.bceloss = nn.BCELoss(reduction="none")
        parameters = (
            list(self.decoder_bin.parameters())
            + list(self.encoder_mean.parameters())
            + list(self.encoder_logv.parameters())
        )
        self.optimizer = optim.Adam(parameters, lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self, z):
        z = torch.FloatTensor(z)
        ret_t = self.decoder_bin(z)
        return torch.bernoulli(ret_t).detach().numpy()

    def propensity_score(self, z, t, sign=1):
        z = torch.FloatTensor(z)
        lh = self.decoder_bin(z).detach().numpy()
        return np.exp(sign * np.sum(t * np.log(lh) + (1 - t) * np.log(1 - lh), axis=1))

    def neg_elbo(self, t):
        z_mean = self.encoder_mean(t)
        z_lv = self.encoder_logv(t)

        std_z = torch.randn(size=z_mean.size()).to(self.device)
        sample_z = std_z * torch.exp(z_lv) + z_mean

        rec_t = self.decoder_bin(sample_z)

        rec_loss = self.bceloss(rec_t, t).sum(1)

        KL_divergence = 0.5 * (
            (-z_lv) * 2 - 1 + torch.exp(2 * z_lv) + z_mean * z_mean
        ).sum(1)
        return (rec_loss + KL_divergence).mean(), rec_loss.mean(), KL_divergence.mean()

    def rec_loss(self, t, nl):
        z_mean = self.encoder_mean(t)
        z_lv = self.encoder_logv(t)

        std_z = torch.randn(size=z_mean.size()).to(self.device)
        sample_z = std_z * torch.exp(z_lv) * nl + z_mean

        rec_t = self.decoder_bin(sample_z)
        rec_loss = self.bceloss(rec_t, t).sum(1)

        return rec_loss.sum()

    def optimize(self, t):
        self.optimizer.zero_grad()
        loss, rec_l, KL_d = self.neg_elbo(t)
        loss.backward()
        self.optimizer.step()
        return loss.item(), rec_l.item(), KL_d.item()

    def infer(self, t, ifn=0):
        z_infer = self.encoder_mean(t)
        z_noise = torch.exp(self.encoder_logv(t))
        z_infer += ifn * torch.randn(size=z_infer.size()).to(self.device) * z_noise
        # print(torch.cat((self.encoder_mean(t_obsx), torch.exp(self.encoder_logv(t_obsx))), 1).detach().numpy())
        z_infer = z_infer.detach().numpy()
        return z_infer


class Bottle:
    def __init__(self, aux_dim, latent_dim, hidden_dim, output_dim, n_layers):
        self.prior_mean = MLP(aux_dim, latent_dim, hidden_dim, n_layers)
        self.decoder = MLP(latent_dim, output_dim, hidden_dim, n_layers, True)
        self.bceloss = nn.BCELoss(reduction="none")
        parameters = list(self.prior_mean.parameters()) + list(
            self.decoder.parameters()
        )
        self.optimizer = optim.Adam(parameters, lr=0.01)

    def neg_elbo(self, t, obs_x):
        prior_m = self.prior_mean(obs_x)

        rec_t = self.decoder(prior_m)
        rec_loss = self.bceloss(rec_t, t).sum(1)

        return rec_loss.mean()

    def optimize(self, t, obs_x):
        self.optimizer.zero_grad()
        loss = self.neg_elbo(t, obs_x)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def infer(self, obs_x):
        z_infer = self.prior_mean(obs_x)
        z_infer = z_infer.detach().numpy()
        return z_infer


class PreNet(nn.Module):
    def __init__(self, x_dim, t_dim, x_hidden_dim, t_hidden_dim, hidden_dim, n_layers):
        super().__init__()
        self.x_net = nn.Linear(x_dim, x_hidden_dim)
        self.t_net = nn.Linear(t_dim, t_hidden_dim)
        self.n_layers = n_layers
        self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        if self.n_layers == 1:
            _fc_list = [nn.Linear(x_hidden_dim + t_hidden_dim, 1)]
        else:
            _fc_list = [nn.Linear(x_hidden_dim + t_hidden_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], 1))
        self.fc = nn.ModuleList(_fc_list)
        self.apply(weights_init)

    def forward(self, x, t):
        x = F.elu(self.x_net(x))
        t = F.elu(self.t_net(t))
        xt = torch.cat((x, t), dim=1)
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                xt = self.fc[c](xt)
            else:
                xt = F.elu(self.fc[c](xt))
        return xt


class PredictTest:
    def __init__(self, x_dim, t_dim, learning_rate, hidden_dim, n_layers):
        self.pre_net = MLP(x_dim + t_dim, 1, hidden_dim, n_layers)
        self.optimizer = optim.Adam(
            self.pre_net.parameters(), lr=learning_rate, weight_decay=0.0000
        )
        self.mseloss = nn.MSELoss(reduction="none")
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

    def optimize(self, x, t, y, w):
        self.optimizer.zero_grad()
        pre_y = self.pre_net(torch.cat((x, t), dim=1))
        loss = (self.mseloss(pre_y, y) * w).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x, t):
        pre_y = self.pre_net(torch.cat((x, t), dim=1))
        pre_y = pre_y.detach().numpy().squeeze()
        return pre_y

    def adjust_lr(self):
        self.scheduler.step()

    def look(self):
        w1 = self.pre_net.fc[0].weight.detach().numpy()
        w2 = self.pre_net.fc[1].weight.detach().numpy()
        w3 = self.pre_net.fc[2].weight.detach().numpy()
        print(np.square(w1).mean())
        print(np.square(w2).mean())
        print(np.square(w3).mean())
        print(w3)
        print(np.sum(np.square(w1), axis=0))
        print(np.sum(np.square(w3.dot(w2.dot(w1))), axis=0))
        print(np.square(w3.dot(w2.dot(w1))).sum())
        tt = np.sum(np.square(w3.dot(w2.dot(w1))), axis=0)
        print(np.sum(tt[:1]), np.sum(tt[1:5]), np.sum(tt[5:]))
