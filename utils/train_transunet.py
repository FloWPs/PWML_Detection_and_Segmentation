import torch
from torch.optim import SGD
from torch.optim import Adam # modif flora
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Additional Scripts
from utils.transunet import TransUNet
from utils.utils import dice_loss
# # from focal_loss.focal_loss import FocalLoss
# from torchvision.ops import sigmoid_focal_loss
from utils.config import cfg

class TransUNetSeg:
    def __init__(self, device):
        self.device = device
        self.model = TransUNet(img_dim=cfg.transunet.img_dim,
                               in_channels=cfg.transunet.in_channels,
                               out_channels=cfg.transunet.out_channels,
                               head_num=cfg.transunet.head_num,
                               mlp_dim=cfg.transunet.mlp_dim,
                               block_num=cfg.transunet.block_num,
                               patch_dim=cfg.transunet.patch_dim,
                               class_num=cfg.transunet.class_num).to(self.device)

        self.criterion = dice_loss
        # # self.criterion = FocalLoss(gamma=0.9, reduction='sum') # Modif loss par flora # pas de paramètres alpha
        # self.criterion = sigmoid_focal_loss
        # self.optimizer = SGD(self.model.parameters(), lr=cfg.learning_rate,
        #                      momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        self.optimizer = Adam(self.model.parameters(), lr=cfg.learning_rate,
                             weight_decay=cfg.weight_decay) # modif optimizer par flora
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True) # modif flora

    def load_model(self, path):
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        self.model.eval()

    def train_step(self, **params):
        self.model.train()

        self.optimizer.zero_grad()
        pred_mask = self.model(params['img'])
        loss = self.criterion(pred_mask, params['mask'])
        # loss = self.criterion(pred_mask, params['mask'], alpha=0.5, gamma=0.8, reduction='mean')
        # print(pred_mask.shape, pred_mask)
        
        loss.backward()
        self.optimizer.step()

        return loss.item(), pred_mask

    def test_step(self, **params):
        self.model.eval()

        pred_mask = self.model(params['img'])
        loss = self.criterion(pred_mask, params['mask'])
        # loss = self.criterion(pred_mask, params['mask'], alpha=0.5, gamma=0.8, reduction='mean')
        # self.scheduler.step(loss)

        return loss.item(), pred_mask
