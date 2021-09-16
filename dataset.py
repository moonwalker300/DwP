import numpy as np
from scipy import stats
import os

class SimDataset:
    def __init__(self, sample_size, x_dim, t_dim, obs_idx, noise_dim, ifnew, name):
        if ifnew:
            # mean和covs是confounder生成的mean和covs
            means = np.zeros(x_dim)
            covs = np.eye(x_dim) * 1.0 + 0.0
            # noise_mean和noise_covs是noise variable生成的mean和covs
            x = np.random.multivariate_normal(means, covs, sample_size)
            noise = np.random.normal(0.0, 1.0, size = [sample_size, noise_dim])
            for i in range(noise_dim):
                noise[:, i] = noise[:, 0]
            thres_hold = 1.8
            mul_cof = 0.2
            logit_cof = 1.0
            cof_t = np.random.rand(x_dim, t_dim) * 2 - 1.0
            prob = x.dot(cof_t) * logit_cof
            # 这两行无所谓，看一下T生成的bias程度而已
            lh = stats.norm.cdf(prob, loc=0, scale=thres_hold)
            print(np.mean(np.sum(lh * np.log(lh) + (1 - lh) * np.log(1 - lh), axis=1)))

            prob += np.random.normal(0, thres_hold, size=prob.shape)
            t = (0 < prob).astype(np.int32)
            cof_y = np.random.normal(0, 1, size=[x_dim, t_dim]) / 2 + cof_t + 1.0
            print(((cof_t * cof_y).sum()) / (np.sqrt((cof_t * cof_t).sum()) * np.sqrt((cof_y * cof_y).sum())))
            tmp = x.dot(cof_y)
            y = np.sum(tmp * t, axis=1) * mul_cof
            eps = 0
            y += np.random.normal(0, eps, size=y.shape)

            t_test = (0 < np.random.normal(0, 1, size=[sample_size, t_dim])).astype(
                np.int32
            )
            y_test = np.sum(tmp * t_test, axis=1) * mul_cof

            x_out_test = np.random.multivariate_normal(means, covs, sample_size)
            noise_out_test = np.random.uniform(0, 1, size = [sample_size, noise_dim])
            for i in range(noise_dim):
                noise_out_test[:, i] = noise_out_test[:, 0]
            t_out_test = ((0 < np.random.normal(0, 1, size=[sample_size, t_dim])) * 0).astype(
                np.int32
            )
            tmp = x_out_test.dot(cof_y)
            y_out_test = np.sum(tmp * t_out_test, axis=1) * mul_cof
            np.save(name + "x.npy", x)
            np.save(name + "t.npy", t)
            np.save(name + "y.npy", y)
            np.save(name + "noise.npy", noise)
            np.save(name + "t_test.npy", t_test)
            np.save(name + "y_test.npy", y_test)
            np.save(name + "noise_out_test.npy", noise_out_test)
            np.save(name + "x_out_test.npy", x_out_test)
            np.save(name + "t_out_test.npy", t_out_test)
            np.save(name + "y_out_test.npy", y_out_test)
        else:
            x = np.load(name + "x.npy")
            t = np.load(name + "t.npy")
            y = np.load(name + "y.npy")
            noise = np.load(name + "noise.npy")
            t_test = np.load(name + "t_test.npy")
            y_test = np.load(name + "y_test.npy")
            noise_out_test = np.load(name + "noise_out_test.npy")
            x_out_test = np.load(name + "x_out_test.npy")
            t_out_test = np.load(name + "t_out_test.npy")
            y_out_test = np.load(name + "y_out_test.npy")
            if (os.path.isfile(name + "adn.npy")):
                adn = np.load(name + "adn.npy")
                adn_out = np.load(name + "adn_out.npy")
            else:
                adn = None
                adn_out = None

        self.x = x
        self.t = t
        self.y = y
        self.noise = noise
        self.t_test = t_test
        self.y_test = y_test
        self.noise_out_test = noise_out_test
        self.x_out_test = x_out_test
        self.t_out_test = t_out_test
        self.y_out_test = y_out_test
        self.obs_idx = obs_idx
        self.adn = None
        self.adn_out = None
    def sigmoid(self, logit):
        return 1 / (1 + np.exp(-logit))

    def getTrainData(self):
        if len(self.obs_idx) > 0:
            obs = np.concatenate([self.x[:, self.obs_idx], self.noise], axis=1)
        else:
            obs = self.noise
        if (not (self.adn is None)):
            obs = obs + self.adn[:, self.x.shape[1] - len(self.obs_idx):] * 0.2
        
        return self.x, self.t, self.y, obs

    def getInTestData(self):
        return self.t_test, self.y_test

    def getOutTestData(self):
        if len(self.obs_idx) > 0:
            obs = np.concatenate(
                [self.x_out_test[:, self.obs_idx], self.noise_out_test], axis=1
            )
        else:
            obs = self.noise_out_test
        if (not (self.adn_out is None)):
            obs = obs + self.adn_out[:, self.x_out_test.shape[1] - len(self.obs_idx):] * 0.2
        return self.x_out_test, self.t_out_test, self.y_out_test, obs
