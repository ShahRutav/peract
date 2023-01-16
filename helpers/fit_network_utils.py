from helpers.network_utils import *
from helpers.network_utils import _set_train

class FITAndFcsNet(nn.Module):

    def __init__(
            self,
            vit: ViT,
            task_model,
            tokenizer,
            visual_transform,
            low_dim_state_len: int,
            input_resolution: List[int],
            filters: List[int],
            kernel_sizes: List[int],
            strides: List[int],
            norm: str = None,
            fc_layers: List[int] = None,
            activation: str = 'relu',
            task_feat_dim = 256,
        ):

        super(FITAndFcsNet, self).__init__()
        self._vit = copy.deepcopy(vit)
        self._task_model = copy.deepcopy(task_model)
        self._tokenizer = tokenizer
        self._visual_transform = visual_transform
        self._input_channels = 64 + low_dim_state_len
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation
        self._fc_layers = [] if fc_layers is None else fc_layers
        self._input_resolution = input_resolution

        self._task_feat_dim = task_feat_dim

    def build(self):
        layers = []
        channels = self._input_channels

        self.conv1 = Conv2DFiLMBlock(
            channels, self._filters[0], self._kernel_sizes[0],
            self._strides[0], norm=None, activation=None)
        self.gamma1 = nn.Linear(self._task_feat_dim, self._filters[0])
        self.beta1 = nn.Linear(self._task_feat_dim, self._filters[0])

        self.conv2 = Conv2DFiLMBlock(
            self._filters[0], self._filters[1], self._kernel_sizes[1],
            self._strides[1], norm=None, activation=None)
        self.gamma2 = nn.Linear(self._task_feat_dim, self._filters[1])
        self.beta2 = nn.Linear(self._task_feat_dim, self._filters[1])

        self.conv3 = Conv2DFiLMBlock(
            self._filters[1], self._filters[2], self._kernel_sizes[2],
            self._strides[2], norm=None, activation=None)
        self.gamma3 = nn.Linear(self._task_feat_dim, self._filters[2])
        self.beta3 = nn.Linear(self._task_feat_dim, self._filters[2])

        self._maxp = nn.AdaptiveMaxPool2d(1)

        channels = self._filters[-1]
        dense_layers = []
        for n in self._fc_layers[:-1]:
            dense_layers.append(
                DenseBlock(channels, n, activation=self._activation))
            channels = n
        dense_layers.append(
            DenseBlock(channels, self._fc_layers[-1]))
        self._fcs = nn.Sequential(*dense_layers)


    def get_optim_param_group(self, lr):
        params = []
        for name, param in self.named_parameters():
            if name.startswith('_task_model'):
                params.append({'params': param, 'lr': 0.0001*lr})
            else:
                params.append({'params': param, 'lr': lr})
        return params

    def train_module(self): # Should handle both: requries_grad and .train()
        for module_name, module in self.named_modules():
            if module_name.startswith('_task_model'):
                _set_train(module, False)
            else:
                _set_train(module, True)
        return self

    def batch_tokenize(self, texts: List[str], context_length: int = 77):
        all_tokens = []
        if type(texts) is not list:
            texts = texts.tolist()
        assert type(texts) == list and type(texts[0]) == str ## has to be a list of language goals
        all_tokens = self._tokenizer(texts, return_tensors='pt', padding=True, truncation=True) # Trests it like multiple sentences
        return all_tokens

    def forward(self, observations, low_dim_ins, lang_goal_desc=None, goal_image=None, video=None):
        rgb_depth = torch.cat([*observations], dim=1)
        x = self._vit(rgb_depth)
        _, _, h, w = x.shape
        low_dim_latents = low_dim_ins.unsqueeze(
            -1).unsqueeze(-1).repeat(1, 1, h, w)
        combined = torch.cat([x, low_dim_latents], dim=1)

        if video is not None:
            bs, num_cam, t, _, _, _ = video.shape
            video = video.reshape(-1, *video.shape[3:])
            video = self._visual_transform(video)
            video = video.reshape(-1, t, *video.shape[1:])
            goal_emb = self._task_model.module.compute_video(video)
            goal_emb = goal_emb.reshape(bs, *goal_emb.shape[1:])
        if lang_goal_desc is not None:
            with torch.no_grad():
                tokens = self.batch_tokenize(lang_goal_desc)
            tokens = {key: val.cuda() for key, val in tokens.items()}
            goal_emb = self._task_model.module.compute_text(tokens) ## TODO: change this to forward pass computation of task_model
        if goal_image is not None:
            bs, num_cam, t, _, _, _ = goal_image.shape
            goal_image = goal_image.reshape(-1, *goal_image.shape[3:])
            goal_image = self._visual_transform(goal_image)
            goal_image = goal_image.reshape(-1, t, *goal_image.shape[1:])
            goal_emb = self._task_model.module.compute_video(goal_image)
            goal_emb = goal_emb.reshape(bs, *goal_emb.shape[1:])
        assert goal_emb.shape[0] == rgb_depth.shape[0], f"Unequal batch_size between goal_emb: {goal_emb.shape} and observations: {rgb_depth.shape}"

        g1 = self.gamma1(goal_emb)
        b1 = self.beta1(goal_emb)
        x = self.conv1(combined, g1, b1)

        g2 = self.gamma2(goal_emb)
        b2 = self.beta2(goal_emb)
        x = self.conv2(x, g2, b2)

        g3 = self.gamma3(goal_emb)
        b3 = self.beta3(goal_emb)
        x = self.conv3(x, g3, b3)

        x = self._maxp(x).squeeze(-1).squeeze(-1)
        return self._fcs(x)
