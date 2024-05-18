# https://github.com/danchern97/RTD_AE/blob/714d1b78c6afa976229a08210e3870db4c69f794/src/rtd.py
#!/usr/bin/python3
import copy
import numpy as np
import torch
import torch.nn as nn
import ripserplusplus as rpp_py

# from gph.python import ripser_parallel


def lp_loss(a, b, p=2):
    return torch.sum(torch.abs(a - b) ** p)


def get_indicies(DX, rc, dim, card):
    dgm = rc["dgms"][dim]
    pairs = rc["pairs"][dim]

    # Retrieve vertices v_a and v_b by picking the ones achieving the maximal
    # distance among all pairwise distances between the simplex vertices
    indices, pers = [], []
    for i in range(len(pairs)):
        s1, s2 = pairs[i]
        if len(s1) == dim + 1 and len(s2) > 0:
            l1, l2 = np.array(s1), np.array(s2)
            i1 = [s1[v] for v in np.unravel_index(np.argmax(DX[l1, :][:, l1]), [len(s1), len(s1)])]
            i2 = [s2[v] for v in np.unravel_index(np.argmax(DX[l2, :][:, l2]), [len(s2), len(s2)])]
            indices += i1
            indices += i2
            pers.append(dgm[i][1] - dgm[i][0])

    # Sort points with distance-to-diagonal
    perm = np.argsort(pers)
    indices = list(np.reshape(indices, [-1, 4])[perm][::-1, :].flatten())

    # Output indices
    indices = indices[: 4 * card] + [0 for _ in range(0, max(0, 4 * card - len(indices)))]
    return list(np.array(indices, dtype=np.compat.long))


def Rips(DX, dim, card, n_threads, engine):
    # Parameters: DX (distance matrix),
    #             dim (homological dimension),
    #             card (number of persistence diagram points, sorted by distance-to-diagonal)
    if dim < 1:
        dim = 1

    if engine == "ripser":
        DX_ = DX.numpy()
        DX_ = (DX_ + DX_.T) / 2.0  # make it symmetrical
        DX_ -= np.diag(np.diag(DX_))
        rc = rpp_py.run("--format distance --dim " + str(dim), DX_)
    elif engine == "giotto":
        rc = ripser_parallel(DX, maxdim=dim, metric="precomputed", collapse_edges=False, n_threads=n_threads)

    all_indicies = []  # for every dimension
    for d in range(1, dim + 1):
        all_indicies.append(get_indicies(DX, rc, d, card))
    return all_indicies


class RTD_differentiable(nn.Module):
    def __init__(self, dim=1, card=50, mode="minimum", n_threads=25, engine="ripser"):
        super().__init__()

        if dim < 1:
            raise ValueError(f"Dimension should be greater than 1. Provided dimension: {dim}")
        self.dim = dim
        self.mode = mode
        self.card = card
        self.n_threads = n_threads
        self.engine = engine

    def forward(self, Dr1, Dr2, immovable=None):
        # inputs are distance matricies
        d, c = self.dim, self.card

        if Dr1.shape[0] != Dr2.shape[0]:
            raise ValueError(f"Point clouds must have same size. Size Dr1: {Dr1.shape} and size Dr2: {Dr2.shape}")

        if Dr1.device != Dr2.device:
            raise ValueError(f"Point clouds must be on the same devices. Device Dr1: {Dr1.device} and device Dr2: {Dr2.device}")

        device = Dr1.device
        # Compute distance matrices
        #         Dr1 = torch.cdist(r1, r1)
        #         Dr2 = torch.cdist(r2, r2)

        Dzz = torch.zeros((len(Dr1), len(Dr1)), device=device)
        if self.mode == "minimum":
            Dr12 = torch.minimum(Dr1, Dr2)
            DX = torch.cat((torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr12), 1)), 0)
            if immovable == 2:
                DX_2 = torch.cat((torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr1), 1)), 0)  # Transfer gradient for edge minimization to edges in cloud #1
            elif immovable == 1:
                DX_2 = torch.cat((torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr2), 1)), 0)  # Transfer gradient from edge minimization to edges in cloud #2
            else:
                DX_2 = DX
        else:
            Dr12 = torch.maximum(Dr1, Dr2)
            DX = torch.cat((torch.cat((Dzz, Dr12.T), 1), torch.cat((Dr12, Dr2), 1)), 0)
            if immovable == 2:
                DX_2 = torch.cat((torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr2), 1)), 0)  # Transfer gradient for edge minimization to edges in cloud #1
            elif immovable == 1:
                DX_2 = torch.cat((torch.cat((Dzz, Dr2.T), 1), torch.cat((Dr2, Dr2), 1)), 0)  # Transfer gradient from edge minimization to edges in cloud #2
            else:
                DX_2 = DX

        # Compute vertices associated to positive and negative simplices
        # Don't compute gradient for this operation
        all_ids = Rips(DX.detach().cpu(), self.dim, self.card, self.n_threads, self.engine)
        all_dgms = []
        for ids in all_ids:
            # Get persistence diagram by simply picking the corresponding entries in the distance matrix
            tmp_idx = np.reshape(ids, [2 * c, 2])
            if self.mode == "minimum":
                dgm = torch.hstack(
                    [torch.reshape(DX[tmp_idx[::2, 0], tmp_idx[::2, 1]], [c, 1]), torch.reshape(DX_2[tmp_idx[1::2, 0], tmp_idx[1::2, 1]], [c, 1])]
                )
            else:
                dgm = torch.hstack(
                    [torch.reshape(DX_2[tmp_idx[::2, 0], tmp_idx[::2, 1]], [c, 1]), torch.reshape(DX[tmp_idx[1::2, 0], tmp_idx[1::2, 1]], [c, 1])]
                )
            all_dgms.append(dgm)
        return all_dgms


class RTDLoss(nn.Module):
    def __init__(self, dim=1, card=50, n_threads=25, engine="ripser", mode="minimum", is_sym=True, lp=1.0, **kwargs):
        super().__init__()

        self.is_sym = is_sym
        self.mode = mode
        self.p = lp
        self.rtd = RTD_differentiable(dim, card, mode, n_threads, engine)

    def forward(self, x_dist, z_dist):
        # x_dist is the precomputed distance matrix
        # z is the batch of latent representations
        loss = 0.0
        loss_xz = 0.0
        loss_zx = 0.0
        rtd_xz = self.rtd(x_dist, z_dist, immovable=1)
        if self.is_sym:
            rtd_zx = self.rtd(z_dist, x_dist, immovable=2)
        for d, rtd in enumerate(rtd_xz):  # different dimensions
            loss_xz += lp_loss(rtd_xz[d][:, 1], rtd_xz[d][:, 0], p=self.p)
            if self.is_sym:
                loss_zx += lp_loss(rtd_zx[d][:, 1], rtd_zx[d][:, 0], p=self.p)
        loss = (loss_xz + loss_zx) / 2.0
        return loss_xz, loss_zx, loss


class MinMaxRTDLoss(nn.Module):
    def __init__(self, dim=1, card=50, n_threads=25, engine="ripser", is_sym=True, lp=1.0, **kwargs):
        super().__init__()

        self.is_sym = is_sym
        self.p = lp
        self.rtd_min = RTD_differentiable(dim, card, "minimum", n_threads, engine)
        self.rtd_max = RTD_differentiable(dim, card, "maximum", n_threads, engine)

    def forward(self, x_dist, z_dist):
        # x_dist is the precomputed distance matrix
        # z is the batch of latent representations
        loss = 0.0
        loss_xz = 0.0
        loss_zx = 0.0
        rtd_xz = self.rtd_min(x_dist, z_dist, immovable=1) + self.rtd_max(x_dist, z_dist, immovable=1)
        if self.is_sym:
            rtd_zx = self.rtd_min(z_dist, x_dist, immovable=2) + self.rtd_max(z_dist, x_dist, immovable=2)
        for d, rtd in enumerate(rtd_xz):  # different dimensions
            loss_xz += lp_loss(rtd_xz[d][:, 1], rtd_xz[d][:, 0], p=self.p)
            if self.is_sym:
                loss_zx += lp_loss(rtd_zx[d][:, 1], rtd_zx[d][:, 0], p=self.p)
        loss = (loss_xz + loss_zx) / 2.0
        return loss_xz, loss_zx, loss


class FeatureMapLoss(nn.Module):
    def __init__(
        self,
        regularizer_model,
        device,
        which_model_is_regularizer="first",
        fmap_betas=[0, 0, 0, 0, 0],
        fmap_loss_type="RTDLoss",
        dist_p=2.0,
        **kwargs,
    ):
        super().__init__()

        self.device = device
        self.regularizer_model = None
        self.is_activated = False
        if which_model_is_regularizer == "pretrain":
            self.copy_if_none_regularizer_model(regularizer_model)
            self.activate()
        self.betas = fmap_betas
        self.loss_type = fmap_loss_type
        print(fmap_loss_type)
        if fmap_loss_type == "RTDLoss":
            self.loss_fn = RTDLoss(**kwargs)
        elif fmap_loss_type == "MinMaxRTDLoss":
            self.loss_fn = MinMaxRTDLoss(**kwargs)
        elif fmap_loss_type == "MSELoss":
            print("!")
            self.loss_fn = torch.nn.MSELoss()
        self.loss_fn = self.loss_fn.to(device)
        self.dist_p = dist_p

    def copy_if_none_regularizer_model(self, regularizer_model):
        if self.regularizer_model is None and not self.is_activated:
            self.regularizer_model = copy.deepcopy(regularizer_model).to(self.device)
            self.regularizer_model.eval()

    def activate(self):
        # activate only if any coefficient is nonzero
        for beta in self.betas:
            if beta != 0:
                self.is_activated = True
                break
        print(f"FeatureMapLoss.is_activated: {self.is_activated}")

    def regularizer_model_forward(self, images):
        if not self.is_activated or self.regularizer_model is None:
            return None
        with torch.no_grad():
            return self.regularizer_model(images, True)[1]

    def forward(self, f_map, regularizer_f_map):
        f_map_loss = torch.tensor([0.0], device=self.device)
        if not self.is_activated or regularizer_f_map is None:
            return f_map_loss, None, None, None

        losses_xz, losses_zx, losses = [], [], []
        for i in range(len(f_map)):
            if self.betas[i] == 0:
                continue
            
            if self.loss_type == "MSELoss":
                loss = self.loss_fn(f_map[i], regularizer_f_map[i])
            else:
                regularizer_dist = torch.cdist(regularizer_f_map[i], regularizer_f_map[i], p=self.dist_p) / np.sqrt(regularizer_f_map[i].shape[1])
                cur_dist = torch.cdist(f_map[i], f_map[i], p=self.dist_p) / np.sqrt(f_map[i].shape[1])
                _, _, loss = self.loss_fn(regularizer_dist, cur_dist)
                # losses_xz.append(loss_xz)
                # losses_zx.append(loss_zx)
                # losses.append(loss)
            f_map_loss = f_map_loss + self.betas[i] * loss

        return f_map_loss, losses_xz, losses_zx, losses
