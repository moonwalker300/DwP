import argparse
from termcolor import colored


import numpy as np
import torch
from torch import optim
import random

from dataset import SimDataset
from model_main import CEVAE, CVAE, IVAE, MLP, TVAE, VAE, PredictTest

class Log:
    def __init__(self, filename):
        self.filename = filename
    def log(self, content):
        with open(self.filename, "a") as f:
            f.write(content + '\n')
            f.close()
        print(content)

def RMSE(pre, target):
    mse = np.mean(np.square(pre - target))
    return np.sqrt(mse)


parser = argparse.ArgumentParser()
parser.add_argument("-l", type=float, default=0.001, help="learning rate")
parser.add_argument("--latdim", type=int, default=5, help="Latent Dimension")
parser.add_argument("--obsm", type=int, default=0, help="Observed Dimension")
parser.add_argument("--mask", type=int, default=0, help="Mask ObsX")
parser.add_argument("--ylayer", type=int, default=50, help="Y Layer Dimension")
parser.add_argument("--nlayer", type=int, default=50, help="N Layer Dimension")
parser.add_argument("--stop", type=int, default=5000, help="Stop Epochs")
args = parser.parse_args()
lr = args.l
latent_dim = args.latdim
stop = args.stop
# np.set_printoptions(threshold=10000)
# np.set_printoptions(suppress=True)
n = 10000  # 样本数量
m = 5  # Confounder维度
p = 20  # Treatment维度
noise_dim = 10  # Observed Noise维度
new_data = False # 是否重新生成Simulation Data
obs_idx = list(range(args.obsm))
# Observed Covariate包括一部分Observed Confounder+Noisy Variable
name = "Obs_confounder5_n10_t20_cor02_logit08_2"  # Simulation Data的名字
data = SimDataset(n, m, p, obs_idx, noise_dim, new_data, name)

x, t, y, obs_x = data.getTrainData()
# x是Confounder的Ground Truth，obs_x是Observed Covariate包括一部分Observed Confounder+Noisy Variable
t_test, y_test = data.getInTestData()  # Insample的Test data
x_out_test, t_out_test, y_out_test, obs_x_out = data.getOutTestData()
# Outsample的，但是现在还没用到它，可以忽略

file_name = name + "_obs" + str(args.obsm)
filelog = Log("dec_res_{}.txt".format(file_name))

filelog.log("Experiment Start!")
filelog.log(str(args))
filelog.log("Y Mean %f, Std %f " % (np.mean(y), np.std(y)))
filelog.log("Test Y Mean %f, Std %f " % (np.mean(y_test), np.std(y_test)))
filelog.log('Observe confounder %d, Noise %d dimension' % (args.obsm, noise_dim))

obsm = obs_x.shape[1]
obs_x[:, obsm - args.mask:] = 0 # 这是是我想Mask掉几维Noise Variable的。可以忽略
obs_x_out[:, obsm - args.mask:] = 0
obsm = obs_x.shape[1]
filelog.log("Learning Rate %f" % (lr))

hidden_size = args.nlayer
n_layers = 3
y_layers = 3
y_hidden = args.ylayer
epochs = 3000
batch_size = 1024
rep_times = 10
m1 = []
m2 = []

rec_conf_dec = []
rec_noise_dec = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def manual_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


for rep in range(rep_times):
    manual_seed(rep + 1)
    filelog.log(colored("========== repeat time {} ==========".format(rep + 1), "red"))

    t_vae = TVAE(latent_dim, hidden_size, p, n_layers).to(device)

    filelog.log(colored("== Deconfounder: Training all ==".format(rep + 1), "blue"))

    last_loss = 100000
    last_epoch = -1
    for ep in range(epochs):
        idx = np.random.permutation(n)
        KL_loss_s = []
        loss_s = []
        t_loss_s = []
        for j in range(0, n, batch_size):
            op, ed = j, min(j + batch_size, n)
            t_batch = torch.FloatTensor(t[idx[op:ed]]).to(device)

            loss, t_loss, KL_loss = t_vae.optimize(t_batch)
            loss_s.append(loss * (ed - op))
            KL_loss_s.append(KL_loss * (ed - op))
            t_loss_s.append(t_loss * (ed - op))
        if (ep + 1) % 50 == 0:
            filelog.log("Epoch %d " % (ep))
            filelog.log("Overall Loss: %f" % (sum(loss_s) / n))
            filelog.log("KL Loss: %f" % (sum(KL_loss_s) / n))
            filelog.log("T Loss: %f" % (sum(t_loss_s) / n))
        current_loss = sum(loss_s) / n
        if current_loss < last_loss:
            last_loss = current_loss
            last_epoch = ep
        if ep - last_epoch > stop:
            break
        # if (ep + 1) % 50 == 0:
        #     if sum(loss_s) / n >= last_loss * 1.00:
        #         break
        #     else:
        #         last_loss = sum(loss_s) / n

    # 这个是我们学到的Confounder表征重构Confounder
    filelog.log(colored("== Deconfounder: Reconstructing confounder ==".format(rep + 1), "blue"))
    rec_net = MLP(latent_dim, x.shape[1], 20, 3).to(device)
    optimizer = optim.Adam(rec_net.parameters(), lr=0.005)
    euc = torch.nn.MSELoss(reduction="none")
    last_loss = 100000
    for ep in range(epochs):
        idx = np.random.permutation(n)
        rec_loss_s = []
        for j in range(0, n, batch_size):
            op, ed = j, min(j + batch_size, n)
            t_batch = torch.FloatTensor(t[idx[op:ed]]).to(device)
            z_batch = t_vae.infer(t_batch, ifn=1)
            x_batch = torch.FloatTensor(x[idx[op:ed]]).to(device)
            pre_x = rec_net(z_batch)
            loss = euc(pre_x, x_batch).sum(1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            rec_loss_s.append(loss.item() * (ed - op))
        if (ep + 1) % 50 == 0:
            filelog.log("Epoch %d" % (ep))
            filelog.log("Rec Loss: %f" % (sum(rec_loss_s) / n))
            if (ep + 1) % 50 == 0:
                if sum(rec_loss_s) / n >= last_loss * 1.00:
                    break
                else:
                    last_loss = sum(rec_loss_s) / n
    rec_conf_dec.append(last_loss)

    # 这个是我们学到的Confounder表征重构Noisy Variable,如果损失越大，越好，如果刚好等于noise_dim，说明与Noisy Variable完全独立
    filelog.log(colored("== Deconfounder: Reconstructing noise ==".format(rep + 1), "blue"))
    rec_net = MLP(latent_dim, obs_x.shape[1] - args.obsm, 20, 3).to(device)
    optimizer = optim.Adam(rec_net.parameters(), lr=0.005)
    euc = torch.nn.MSELoss(reduction="none")
    last_loss = 100000
    for ep in range(epochs):
        idx = np.random.permutation(n)
        rec_loss_s = []
        for j in range(0, n, batch_size):
            op, ed = j, min(j + batch_size, n)
            obsx_batch = torch.FloatTensor(obs_x[idx[op:ed]]).to(device)
            t_batch = torch.FloatTensor(t[idx[op:ed]]).to(device)
            z_batch = t_vae.infer(t_batch, ifn=1)

            pre_x = rec_net(z_batch)
            loss = euc(pre_x, obsx_batch[:, args.obsm:]).sum(1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            rec_loss_s.append(loss.item() * (ed - op))
        if (ep + 1) % 50 == 0:
            filelog.log("Epoch %d" % (ep))
            filelog.log("Rec Loss: %f" % (sum(rec_loss_s) / n))
            if (ep + 1) % 50 == 0:
                if sum(rec_loss_s) / n >= last_loss * 1.00:
                    break
                else:
                    last_loss = sum(rec_loss_s) / n
    rec_noise_dec.append(last_loss)

    filelog.log(colored("== Deconfounder Regression: Training all ==".format(rep + 1), "blue"))
    rec_z = np.zeros([n, latent_dim])
    for i in range(0, n, batch_size):
        op, ed = i, min(i + batch_size, n)
        t_batch = torch.FloatTensor(t[op:ed]).to(device)
        rec_z[op:ed] = t_vae.infer(t_batch, ifn = 0).detach().cpu().numpy()

    infer_x = rec_z.copy()
    pre_test = PredictTest(infer_x.shape[1], p, 
            learning_rate=0.005, hidden_dim=y_hidden, n_layers=y_layers).to(device)

    last_loss = 100000
    for ep in range(epochs * 2):
        idx = np.random.permutation(n)
        pre_loss_s = []
        for j in range(0, n, batch_size):
            op, ed = j, min(j + batch_size, n)
            x_batch = torch.FloatTensor(infer_x[idx[op:ed]]).to(device)
            t_batch = torch.FloatTensor(t[idx[op:ed]]).to(device)
            y_batch = torch.FloatTensor(y[idx[op:ed]]).view(-1, 1).to(device)
            pre_loss = pre_test.optimize(x_batch, t_batch, y_batch)
            pre_loss_s.append(pre_loss * (ed - op))
        if (ep + 1) % 500 == 0:
            filelog.log("Epoch %d " % (ep))
            filelog.log("Prediction Loss: %f" % (sum(pre_loss_s) / n))
        current_loss = sum(pre_loss_s) / n
        if current_loss < last_loss:
            last_loss = current_loss
            last_epoch = ep
        if ep - last_epoch > stop:
            break
    filelog.log(colored("== Deconfounder Regression: Testing in sample performance ==".format(rep + 1), "blue"))
    Train_y = np.zeros(n)
    for i in range(0, n, batch_size):
        op, ed = i, min(i + batch_size, n)
        x_batch = torch.FloatTensor(infer_x[op:ed]).to(device)
        t_batch = torch.FloatTensor(t[op:ed]).to(device)
        Train_y[op:ed] = pre_test.predict(x_batch, t_batch)

    # In-sample
    Insample_y = np.zeros(n)
    for i in range(0, n, batch_size):
        op, ed = i, min(i + batch_size, n)
        x_batch = torch.FloatTensor(infer_x[op:ed]).to(device)
        t_batch = torch.FloatTensor(t_test[op:ed]).to(device)
        Insample_y[op:ed] = pre_test.predict(x_batch, t_batch)
    filelog.log("Train Error %f" % (RMSE(Train_y, y)))
    filelog.log("Insample Error %f" % (RMSE(Insample_y, y_test)))
    m1.append(RMSE(Insample_y, y_test))
    
    filelog.log(colored("== Deconfounder plus Regression: Training all ==".format(rep + 1), "blue"))
    infer_x = np.concatenate([rec_z, obs_x], axis = 1)
    pre_test = PredictTest(infer_x.shape[1], p,
            learning_rate=0.005, hidden_dim=y_hidden, n_layers=y_layers).to(device)

    last_loss = 100000
    for ep in range(epochs * 2):
        idx = np.random.permutation(n)
        pre_loss_s = []
        for j in range(0, n, batch_size):
            op, ed = j, min(j + batch_size, n)
            x_batch = torch.FloatTensor(infer_x[idx[op:ed]]).to(device)
            t_batch = torch.FloatTensor(t[idx[op:ed]]).to(device)
            y_batch = torch.FloatTensor(y[idx[op:ed]]).view(-1, 1).to(device)
            pre_loss = pre_test.optimize(x_batch, t_batch, y_batch)
            pre_loss_s.append(pre_loss * (ed - op))
        if (ep + 1) % 500 == 0:
            filelog.log("Epoch %d " % (ep))
            filelog.log("Prediction Loss: %f" % (sum(pre_loss_s) / n))
        current_loss = sum(pre_loss_s) / n
        if current_loss < last_loss:
            last_loss = current_loss
            last_epoch = ep
        if ep - last_epoch > stop:
            break
    filelog.log(colored("== Deconfounder plus Regression: Testing in sample performance ==".format(rep + 1), "blue"))
    Train_y = np.zeros(n)
    for i in range(0, n, batch_size):
        op, ed = i, min(i + batch_size, n)
        x_batch = torch.FloatTensor(infer_x[op:ed]).to(device)
        t_batch = torch.FloatTensor(t[op:ed]).to(device)
        Train_y[op:ed] = pre_test.predict(x_batch, t_batch)
    # In-sample
    Insample_y = np.zeros(n)
    for i in range(0, n, batch_size):
        op, ed = i, min(i + batch_size, n)
        x_batch = torch.FloatTensor(infer_x[op:ed]).to(device)
        t_batch = torch.FloatTensor(t_test[op:ed]).to(device)
        Insample_y[op:ed] = pre_test.predict(x_batch, t_batch)
    filelog.log("Train Error %f" % (RMSE(Train_y, y)))
    filelog.log("Insample Error %f" % (RMSE(Insample_y, y_test)))
    m2.append(RMSE(Insample_y, y_test)) 

m1 = np.array(m1)
m2 = np.array(m2)
rec_conf_dec = np.array(rec_conf_dec)
rec_conf_dec = rec_conf_dec[~np.isnan(rec_conf_dec)]
rec_noise_dec = np.array(rec_noise_dec)
rec_noise_dec = rec_noise_dec[~np.isnan(rec_noise_dec)]

filelog.log("Deconfounder, Insample RMSE")
for i in range(m1.shape[0]):
    filelog.log("%.4f, " % m1[i])
filelog.log("Deconfounder plus, Insample RMSE")
for i in range(m2.shape[0]):
    filelog.log("%.4f, " % m2[i])

output = ""
try:
    output += "Dec, RMSE mean %.4f std %.4f, reconstruct confounder %.4f (%.4f) noise %.4f (%.4f)\n" % (
        m1.mean(),
        m1.std(),
        np.mean(rec_conf_dec),
        np.std(rec_conf_dec),
        np.mean(rec_noise_dec),
        np.std(rec_noise_dec),
    )

except:
    pass
try:
    output += "Dec plus, RMSE mean %.4f std %.4f\n" % (
        m2.mean(),
        m2.std(),
    )

except:
    pass

filelog.log(output)
