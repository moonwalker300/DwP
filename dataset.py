import numpy as np
from scipy import stats


class SimDataset:
    def __init__(self, sample_size, x_dim, t_dim, obs_idx, noise_dim, ifnew, name):
        if ifnew:
            # mean和covs是confounder生成的mean和covs
            means = np.zeros(x_dim)
            covs = np.eye(x_dim) * 0.5 + 0.5
            # noise_mean和noise_covs是noise variable生成的mean和covs
            noise_mean = np.zeros(noise_dim)
            noise_covs = np.eye(noise_dim) * 1.0
            for i in range(noise_dim - 1):
                noise_covs[i][i + 1] = 0.5
                noise_covs[i + 1][i] = 0.5
            noise = np.random.multivariate_normal(noise_mean, noise_covs, sample_size)
            thres_hold = 1.8
            mul_cof = 0.25
            logit_cof = 0.4
            x = np.random.multivariate_normal(means, covs, sample_size)
            cof_t = np.random.rand(x_dim, t_dim)
            prob = x.dot(cof_t) * logit_cof
            # 这两行无所谓，看一下T生成的bias程度而已
            lh = stats.norm.cdf(prob, loc=0, scale=thres_hold)
            print(np.mean(np.sum(lh * np.log(lh) + (1 - lh) * np.log(1 - lh), axis=1)))

            prob += np.random.normal(0, thres_hold, size=prob.shape)
            t = (0 < prob).astype(np.int32)
            cof_y = np.random.normal(0, 1, size=[x_dim, t_dim]) / 2 + cof_t

            tmp = x.dot(cof_y)
            y = np.sum(tmp * t, axis=1) * mul_cof
            eps = 0
            y += np.random.normal(0, eps, size=y.shape)

            t_test = (0 < np.random.normal(0, 1, size=[sample_size, t_dim])).astype(
                np.int32
            )
            y_test = np.sum(tmp * t_test, axis=1) * mul_cof

            x_out_test = np.random.multivariate_normal(means, covs, sample_size)
            t_out_test = (0 < np.random.normal(0, 1, size=[sample_size, t_dim])).astype(
                np.int32
            )
            noise_out_test = np.random.multivariate_normal(
                noise_mean, noise_covs, sample_size
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

    def sigmoid(self, logit):
        return 1 / (1 + np.exp(-logit))

    def getTrainData(self):
        if len(self.obs_idx) > 0:
            obs = np.concatenate([self.x[:, self.obs_idx], self.noise], axis=1)
        else:
            obs = self.noise
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
        return self.x_out_test, self.t_out_test, self.y_out_test, obs
