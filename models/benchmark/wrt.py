'''Wireless spectrum sensing Transformer designed by Weishan, modified from Visual Transformer'''
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import torch.utils.benchmark as benchmark
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import utils

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.): # dim=64 by default
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1) #use chunk to split into 3 tensors
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
            #FeedForward: dim => mlp_dim => dim
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

'''
v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)'''
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            # should get ( batch, num_patch, patch_width*patch_hight )
            nn.Linear(patch_dim, dim),  #? linear layer processing one patch?
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


'''
#Example of using WrT:
v = WrT(
    spectra_size = (1, 512),
    patch_size = (1,64),
    num_bands = 8,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 1024,
    dropout = 0.1,
    emb_dropout = 0.1
).to(device)
'''
class Net(nn.Module):
    '''Wireless spectrum sensing Transformer'''
    def __init__(self, *, spectra_size, patch_size, num_bands, dim, depth=6, heads=8, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        '''
        spectra_size = (1, 512), should be tuple or it will be converted into format (xx, xx) via "pair"
        patch_size = (1, 64), like spectra_size
        dim = 64, (spectra for one band)
        depth = 6, num of transformer blocks
        mlp_dim = 512, dim of mlp in each transformer block

        '''
        # Consider not using "pair" function
        spectra_height, spectra_width = pair(spectra_size) # should be 1 x 512 = 1x(8*64)
        patch_height, patch_width = pair(patch_size) # want 64-dim vector for each "word"/"patch"
        assert spectra_height % patch_height == 0 and spectra_width % patch_width == 0, 'Original spectra dimensions must be divisible by the patch(band) size.'

        num_patches = (spectra_height // patch_height) * (spectra_width // patch_width)
        patch_dim = channels * patch_height * patch_width #pixes per patch
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim), #linear layer projecting one band spectra
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) # learnable position embedding?
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # learnable input class embedding?
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        #dim of Transformer mlp: dim--mlp_dim--dim
        self.pool = pool
        self.to_latent = nn.Identity() #??doesn't change input? position holder for future config
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim), # learnable norm layer
            nn.Linear(dim, num_bands)
        )

        self._name = f"wrt_L{depth}_D{dim}_C{spectra_size[1] // 2}_F{patch_size[0]}"

    def forward(self, spectra):
        spectra = spectra.unsqueeze(1).float()
        x = self.to_patch_embedding(spectra) # segment the whole spectra and single layer linear projection
        # x should be (batchsize, num_bands, band_spectra) or briefly (b, n, dim)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # cls_token is (1,1,dim) ==> (batch,1,dim) via copying the orginal
        x = torch.cat((cls_tokens, x), dim=1)
        # cat a copy of the learnable class embedding token before(above) other bands
        # x become (b, n+1, dim)
        x += self.pos_embedding[:, :(n + 1)] # (n+1) needed? embed each band & for different batches embed the same-thing
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # according to self.pool, select average pooling or only taking the learned class embedding
        # why not using the whole encoded vector?
        x = self.to_latent(x) # do not change anything
        return self.mlp_head(x)

    def name(self) -> str:
        return self._name


class Loss(nn.Module):
    def __init__(self) -> None:
        super(Loss, self).__init__()
        # self.bce_loss = torchvision.ops.sigmoid_focal_loss
        self.bce_loss = nn.BCELoss(reduction="none")

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        # return self.bce_loss(pred, gt.float(), alpha=0.25).sum(dim=-1).mean()
        return self.bce_loss(torch.sigmoid(pred), gt.float()).sum(dim=-1).mean()

    def acc(self, pred: Tensor, gt: Tensor):
        pred = (pred > 0).long()
        shot = torch.all(gt == pred, dim=-1).sum().item()
        total = gt.size(0)

        return total, shot

    def stat(self, pred: Tensor, gt: Tensor):
        # pred = (pred > 0).long()
        pred = (torch.sigmoid(pred) > 0.8).long()
        fp = torch.sum((pred == 1) & (gt == 0)).item()
        fn = torch.sum((pred == 0) & (gt == 1)).item()
        positives = (gt == 1).sum().item()
        negatives = (gt == 0).sum().item()

        return fp, fn, positives, negatives


class WrTOct26(nn.Module):
    '''Wireless spectrum sensing Transformer'''
    def __init__(self, *, spectra_size, patch_size, num_bands, dim, depth=6, heads=8, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        '''
        spectra_size = (1, 512), should be tuple or it will be converted into format (xx, xx) via "pair"
        patch_size = (1, 64), like spectra_size
        dim = 64, (spectra for one band)
        depth = 6, num of transformer blocks
        mlp_dim = 512, dim of mlp in each transformer block

        '''
        # Consider not using "pair" function
        spectra_height, spectra_width = pair(spectra_size) # should be 1 x 512 = 1x(8*64)
        patch_height, patch_width = pair(patch_size) # want 64-dim vector for each "word"/"patch"
        assert spectra_height % patch_height == 0 and spectra_width % patch_width == 0, 'Original spectra dimensions must be divisible by the patch(band) size.'

        num_patches = (spectra_height // patch_height) * (spectra_width // patch_width)
        patch_dim = channels * patch_height * patch_width #pixes per patch
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim), #linear layer projecting one band spectra
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) # learnable position embedding?
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # learnable input class embedding?
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        #dim of Transformer mlp: dim--mlp_dim--dim
        self.pool = pool
        self.to_latent = nn.Identity() #??doesn't change input? position holder for future config
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim), # learnable norm layer
            nn.Linear(dim, num_bands)
        )
        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim*(num_patches+1)), # learnable norm layer
            nn.Linear(dim*(num_patches+1), num_bands) #Should consider patch !=64
        )

    def forward(self, spectra: torch.Tensor):
        x = self.to_patch_embedding(spectra) # segment the whole spectra and single layer linear projection
        # x should be (batchsize, num_bands, band_spectra) or briefly (b, n, dim)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # cls_token is (1,1,dim) ==> (batch,1,dim) via copying the original
        x = torch.cat((cls_tokens, x), dim=1)
        # cat a copy of the learnable class embedding token before(above) other bands
        # x become (b, n+1, dim)
        x += self.pos_embedding[:, :(n + 1)] # (n+1) needed? embed each band & for different batches embed the same-thing
        x = self.dropout(x)
        x = self.transformer(x)
        # print('x size:', x.size())
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # print('Rearrange dim', rearrange(x,'b p d -> b (p d)').size() )
        # print('x mean size', x.size())
        # according to self.pool, select average pooling or only taking the learned class embedding
        # why not using the whole encoded vector?
        x = self.to_latent(x) # do not change anything

        return self.mlp_head1(rearrange(x, 'b p d -> b (p d)'))


H = 6.626e-34
def train(model: Net, loss_fn: Loss, train_dl: DataLoader, optimizer: Optimizer, scheduler):
    model.train()
    train_loss = 0
    train_shot = 0
    train_total = 0
    print("learning rate:", optimizer.param_groups[0]["lr"])
    for n, (x, gt) in enumerate(train_dl):
        x = x.cuda().float()
        x = utils.add_noise(x)
        gt = gt.cuda()
        gt = (gt > -1).long()

        pred = model(x)
        loss = loss_fn(pred, gt)
        train_loss = train_loss * n / (n + 1) + loss.item() / (n + 1)
        stats = loss_fn.acc(pred, gt)

        total, shot = stats
        train_shot += shot
        train_total += total

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step(train_loss)

    train_acc = round(train_shot / (train_total + H), 4)
    print(
        ".......train loss {}, acc {}".format(train_loss, train_acc), flush=True
    )

def test(model: Net, loss_fn: Loss, test_dl: DataLoader, snr: float | None = 10):
    model.eval()
    test_loss = 0
    test_shot = 0
    test_total = 0
    false_positive = 0
    false_negative = 0
    positive = 0
    negative = 0
    with torch.no_grad():
        for n, (x, gt) in enumerate(test_dl):
            x = x.cuda().float()
            x = utils.add_noise(x, snr)
            gt = gt.cuda()
            gt = (gt > -1).long()
            pred = model(x)
            loss = loss_fn(pred, gt)
            test_loss = test_loss * n / (n + 1) + loss.item() / (n + 1)

            stats = loss_fn.acc(pred, gt)
            total, shot = stats
            test_shot += shot
            test_total += total

            fp, fn, pos, neg = loss_fn.stat(pred, gt)
            false_positive += fp
            false_negative += fn
            positive += pos
            negative += neg

    test_acc = round(test_shot / (test_total + H), 4)
    print(
        ".......test loss {}, acc {}".format(test_loss, test_acc), flush=True
    )
    print("false alarm:", false_positive / (negative + H))
    print("miss detection:", false_negative / (positive + H))

def timeit(model: Net, test_dl: DataLoader, runs: int = 100):
    with torch.no_grad():
        for (x, _) in test_dl:
            x = x.cuda().float()
            timer = benchmark.Timer(
                stmt="model(x)",
                globals={"model": model, "x": x},
            )
            results = [timer.timeit(1).mean for _ in range(runs)]
            print(f"{runs} runs, batch size {x.size(0)}, mean {np.mean(results) * 1000} ms, std {np.std(results) * 1000} ms")
            return


def test_params():
    n_coset = 5
    d_model = 128
    model = Net(
        spectra_size = (2400, 2 * n_coset),
        patch_size = (16, 2 * n_coset),
        num_bands = 16,
        dim = d_model,
        depth = 3,
        heads = 4,
        mlp_dim = d_model * 4,
        dropout = 0.1,
        emb_dropout = 0.1
    ).cuda()

    import torchinfo
    torchinfo.summary(model, input_size=(3, 2400, 2 * n_coset))

    model = model.cuda()
    model.eval()
    input = torch.randn(1024, 2400, 2 * n_coset).cuda()
    import torch.utils.benchmark as benchmark
    with torch.no_grad():
        timer = benchmark.Timer(
            stmt="model(input)",
            globals={"model": model, "input": input},
        )
        print(timer.timeit(10))

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    test_params()