#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import torch
from torch import nn
import torchvision
import torch.nn.functional as F

from .src.resnet import resnet9
from .src.transformer import TransformerPooling
from .src.vmz import r2plus1d_18, r2plus1d_34


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class TransposeSqueeze(nn.Module):
    def __init__(self, fdim, tdim):
        super(TransposeSqueeze, self).__init__()
        self.fdim = fdim
        self.tdim = tdim

    def forward(self, x):
        return x.view(-1, self.fdim, self.tdim).transpose(-1,-2)


class Flatten(nn.Module):
    """A shape adaptation layer to patch certain networks."""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Unsqueeze(nn.Module):
    """A shape adaptation layer to patch certain networks."""
    def __init__(self):
        super(Unsqueeze, self).__init__()

    def forward(self, x):
        return x.unsqueeze(-1)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


def random_weight_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None: 
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class MLP(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.3):
        super(MLP, self).__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes, bias=True)
            )
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                Unsqueeze(),
                nn.BatchNorm1d(n_hidden),
                Flatten(),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True)
            )

    def forward(self, x):
        return self.block_forward(x)


def get_video_feature_extractor(
    vid_base_arch='r2plus1d_18', 
    pretrained=False, 
    duration=1, 
    pre_pool=False, 
):
    if vid_base_arch == 'r2plus1d_18':
        model = r2plus1d_18(pretrained=pretrained, larger_last=False)
        if not pretrained:
            print("Randomy initializing models")
            random_weight_init(model)
        if pre_pool:
            model.avgpool = nn.Identity()
        else:
            model.avgpool = nn.AdaptiveAvgPool3d((duration, 1, 1))
    elif vid_base_arch == 'r2plus1d_34':
        model = r2plus1d_34(pretrained=pretrained)
        if not pretrained:
            print("Randomy initializing models")
            random_weight_init(model)
        if pre_pool:
            model.avgpool = nn.Identity()
        else:
            model.avgpool = nn.AdaptiveAvgPool3d((duration, 1, 1))
    model.fc = Identity()
    return model


def get_audio_feature_extractor(
    aud_base_arch='resnet18', 
    pretrained=False, 
    duration=1
):
    assert(aud_base_arch in ['resnet9', 'resnet18'])
    if aud_base_arch == 'resnet18':
        model = torchvision.models.__dict__[aud_base_arch](
            pretrained=pretrained)
        model.conv1 = torch.nn.Conv2d(
            1, 
            64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False
        )
        model.fc = Identity()
        return model
    elif aud_base_arch == 'resnet9':
        print('resnet9, duration:', duration)
        model = resnet9(pretrained=False,progress=False)

        model.conv1 = torch.nn.Conv2d(
            1, 
            64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False
        )
        model.avgpool = nn.AdaptiveAvgPool2d((1,duration)) 
        return model


class VideoBaseNetwork(nn.Module):
    def __init__(
        self, 
        vid_base_arch='r2plus1d_18', 
        pretrained=False, 
        norm_feat=False, 
        duration=1, 
        pre_pool=False,
    ):
        super(VideoBaseNetwork, self).__init__()
        self.base = get_video_feature_extractor(
            vid_base_arch, 
            pretrained=pretrained,
            duration=duration,
            pre_pool=pre_pool,
        )
        self.norm_feat = norm_feat

    def forward(self, x):
        x = self.base(x).squeeze()
        if self.norm_feat:
            x = F.normalize(x, p=2, dim=1)
        return x


class AudioBaseNetwork(nn.Module):
    def __init__(
        self, 
        aud_base_arch='resnet9', 
        pretrained=False, 
        norm_feat=False, 
        duration=1
    ):
        super(AudioBaseNetwork, self).__init__()
        self.base = get_audio_feature_extractor(
            aud_base_arch, 
            pretrained=pretrained,
            duration=duration
        )
        self.norm_feat = norm_feat

    def forward(self, x):
        x = self.base(x).squeeze()
        if self.norm_feat:
            x = F.normalize(x, p=2, dim=1)
        return x


class GDT(nn.Module):
    def __init__(
        self,
        vid_base_arch='r2plus1d_18', 
        aud_base_arch='resnet9',
        pretrained=False, 
        norm_feat=True, 
        use_mlp=False,
        num_classes=256, 
    ):
        super(GDT, self).__init__()
        print('Using GDT model')

        encoder_dim = 512
        encoder_dim_a = 512
        n_hidden = 512

        # Save proprties
        self.use_mlp = use_mlp
        self.norm_feat = norm_feat
        self.encoder_dim = encoder_dim

        self.video_network = VideoBaseNetwork(
            vid_base_arch, 
            pretrained=pretrained
        )
        self.audio_network = AudioBaseNetwork(
            aud_base_arch, 
            pretrained=pretrained
        )

        if use_mlp:
            print("Using MLP projection layer")
            self.mlp_v = MLP(
                encoder_dim, num_classes, n_hidden=n_hidden)
            self.mlp_a = MLP(encoder_dim_a, num_classes)
        else:
            print("Using Linear Layer")
            self.mlp_v = nn.Linear(encoder_dim, num_classes)
            self.mlp_a = nn.Linear(encoder_dim_a, num_classes)


    def forward(self, img, spec, whichhead=0):
        img_features = self.video_network(img).squeeze()
        aud_features = self.audio_network(spec).squeeze()

        if len(aud_features.shape) == 1:
            aud_features = aud_features.unsqueeze(0)
        if len(img_features.shape) == 1:
            img_features = img_features.unsqueeze(0)

        nce_img_features = self.mlp_v(img_features)
        nce_aud_features = self.mlp_a(aud_features)
        if self.norm_feat:
            nce_img_features = F.normalize(
                nce_img_features, p=2, dim=1)
            nce_aud_features = F.normalize(
                nce_aud_features, p=2, dim=1)
        return nce_img_features, nce_aud_features


class Stica_TransformerFMCrop(nn.Module):
    def __init__(
        self,
        vid_base_arch='r2plus1d_18',
        aud_base_arch='resnet9',
        pretrained=False,
        norm_feat=True,
        use_mlp=False,
        num_classes=256,
        args=None,
    ):
        super(Stica_TransformerFMCrop, self).__init__()
        print('Using Stica-Transformer model that enables featuremap returns')
        
        # Save proprties
        self.use_mlp = use_mlp
        self.norm_feat = norm_feat
        encoder_dim_a = 512
        encoder_dim = 512
        self.encoder_dim = encoder_dim
        self.n_hidden = 512
        self.dp = args.dp if args is not None else 0.0
        self.num_layer = args.num_layer if args is not None else 2
        self.num_head = args.num_head if args is not None else 4
        self.positional_emb = args.positional_emb if args else False
        self.qkv_mha = args.qkv_mha if args else False

        if args.num_sec == 1:
            self.duration = 4
            aud_duration = 4
        elif args.num_sec == 2:
            self.duration = 8
            aud_duration = 7
        elif args.num_sec == 3:
            self.duration = 12
            aud_duration=10
        elif args.num_sec == 4:
            self.duration = 15
            aud_duration = 13
        else:
            assert(0)

        # Backbone
        self.video_network = VideoBaseNetwork(
            vid_base_arch,
            pretrained=pretrained,
            duration=self.duration,
            pre_pool=True,
        )
        self.audio_network = AudioBaseNetwork(
            aud_base_arch,
            pretrained=pretrained,
            duration=1
        )

        # Aggregation module
        if args.num_layer > 0:
            print('Using Transformer Pooling')
            transformer = TransformerPooling(
                emb_dim=encoder_dim,
                hidden_dim=encoder_dim,
                num_layer=self.num_layer,
                dp=self.dp,
                num_head=self.num_head,
                positional_emb=self.positional_emb,
                qkv_mha=self.qkv_mha,
            )
            self.video_pooling = nn.Sequential(
                nn.AdaptiveAvgPool3d((args.transformer_time_dim, 1, 1)),
                TransposeSqueeze(fdim=self.n_hidden, tdim=self.duration),
                transformer
            )
        else:
            print('Using Average Pooling')
            self.video_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))

        # FNN module
        if use_mlp:
            print("Using MLP projection layer")
            self.mlp_v = MLP(encoder_dim, num_classes, n_hidden=self.n_hidden)
            self.mlp_a = MLP(encoder_dim_a, num_classes)
        else:
            print("Using Linear Layer")
            self.mlp_v = nn.Linear(encoder_dim, num_classes)
            self.mlp_a = nn.Linear(encoder_dim_a, num_classes)

    def feat2nce(self, img):
        img = self.video_pooling(img).squeeze()
        img = img.view(-1, self.encoder_dim)
        img = self.mlp_v(img)
        img = F.normalize(img, p=2, dim=1)
        return img

    def forward(self, img, spec, params=None):
        ## B: batch size
        ## N: num of chunk (num of sec)
        ## C: num of channel
        ## L: num of input frames for video per chunk
        ## H: height
        ## W: width
        ## T: num of input windows for audio per chunk
        ## S: num of bank (spectrogram)

        # Run backbone architecture
        # B C LN H W => B H V
        img = self.video_network(img).squeeze()
        # B C S TN => B H A
        spec = self.audio_network(spec).squeeze()

        # Feature Cropping Layer
        if params is not None:
            # params = [ space , crops]
            # space = [[ largecrop_locations], [small_croplocations]
            # location = [xmin,xmax,ymin,ymax] or [tmin,tmax]
            crop_nces = [[],[]]
            tcrop_nces = [[],[]]
            s_large_crops, s_small_crops = len(params[0][0]),len(params[0][1])
            t_large_crops, t_small_crops = len(params[1][0]),len(params[1][1])
            for i in range(s_large_crops):
                xmin, xmax, ymin, ymax = params[0][0][i]
                crop_nces[0].append(self.feat2nce(img[..., xmin:xmax,ymin:ymax]))
                for j in range(s_small_crops):
                    xmin, xmax, ymin, ymax = params[0][1][j]
                    crop_nces[1].append(self.feat2nce(img[..., xmin:xmax,ymin:ymax]))
            for ti in range(t_large_crops):
                tmin,tmax= params[1][0][ti]
                tcrop_nces[0].append(self.feat2nce(img[:,:, tmin:tmax, :,:]))
                for tj in range(t_small_crops):
                    tmin,tmax= params[1][1][tj]
                    tcrop_nces[1].append(self.feat2nce(img[:,:, tmin:tmax, :,:]))
        
        # Temporal Pooling: B V H => B H
        img = self.video_pooling(img) 

        # Reshape Layer
        if len(spec.shape) == 1:
            spec = spec.unsqueeze(0)
        img = img.view(-1, self.encoder_dim)

        # MLP projection layer
        img = self.mlp_v(img)
        spec = self.mlp_a(spec)

        # Normalization layer
        if self.norm_feat:
            img = F.normalize(img, p=2, dim=1)
            spec = F.normalize(spec, p=2, dim=1)

        return (img, [crop_nces, tcrop_nces], spec)


class Text_Encoder(nn.Module):
    def __init__(
        self,
        embd_dim=256,
        token_to_word_path='datasets/data/dict.npy',
        num_embeddings=66250,
        word_embedding_dim=300,
        word2vec_path='datasets/data/word2vec.pth',
        max_words=16,
        output_dim=2048
    ):
        super(Text_Encoder, self).__init__()
        if word2vec_path:
            self.word_embd = nn.Embedding.from_pretrained(torch.load(word2vec_path)) 
        else:
            self.word_embd = nn.Embedding(num_embeddings, word_embedding_dim)
        self.fc1 = nn.Linear(word_embedding_dim, output_dim)
        self.word_to_token = {}
        self.max_words = max_words
        token_to_word = np.load(token_to_word_path)
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = torch.zeros(size - len(tensor)).long()
            return torch.cat((tensor, zero), dim=0)

    def is_cuda(self):
        return self.fc1.bias.is_cuda

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(torch.LongTensor(words), self.max_words)
            return we
        else:
            return torch.zeros(self.max_words).long()

    def words_to_ids(self, x):
        split_x = [self._words_to_token(self._split_text(sent)) for sent in x]
        return torch.stack(split_x, dim=0)

    def forward(self, x, raw_text=False):
        if raw_text:
            x = self.words_to_ids(x)
        with torch.no_grad():
            x = self.word_embd(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = torch.max(x, dim=1)[0]
        return x


class TextVid_GDT(nn.Module):
    def __init__(
        self,
        vid_base_arch='r2plus1d_18', 
        text_base_arch='word2vec',
        pretrained=False, 
        norm_feat=True, 
        use_mlp=True,
        num_classes=256, 
    ):
        super(TextVid_GDT, self).__init__()
        print('Using GDT video-text model')

        encoder_dim = 512
        n_hidden = 512
        
        # Save proprties	
        self.use_mlp = use_mlp	
        self.norm_feat = norm_feat
        self.encoder_dim = encoder_dim

        # Backbone architectures
        self.video_network = VideoBaseNetwork(
            vid_base_arch, 
            pretrained=pretrained
        )
        self.text_network = Text_Encoder()

        # Projection Layer
        if use_mlp:
            print("Using MLP projection layer")
            self.mlp_v = MLP(encoder_dim, num_classes)
            self.mlp_t = MLP(2048, num_classes)
        else:
            print("Using Linear Layer")
            self.mlp_v = nn.Linear(512, num_classes)
            self.mlp_t = nn.Linear(2048, num_classes)

    def forward(self, img, text, whichhead=0):
        img_features = self.video_network(img).squeeze()
        text_features = self.text_network(text).squeeze()

        if len(text_features.shape) == 1:
            text_features = text_features.unsqueeze(0)
        if len(img_features.shape) == 1:
            img_features = img_features.unsqueeze(0)

        nce_img_features = self.mlp_v(img_features)
        nce_text_features = self.mlp_t(text_features)
        if self.norm_feat:
            nce_img_features = F.normalize(
                nce_img_features, p=2, dim=1)
            nce_text_features = F.normalize(
                nce_text_features, p=2, dim=1)
        return (
            nce_img_features, nce_text_features, 
        )


def load_model(
    model_type='stica',
    vid_base_arch='r2plus1d_18', 
    aud_base_arch='resnet9',
    pretrained=False, 
    norm_feat=True, 
    use_mlp=False,
    num_classes=256, 
    args=None,
):  
    # Cross-modal GDT
    if model_type == 'stica':
        print('Using Stica-Transformer FM CROP')
        model = Stica_TransformerFMCrop(
            vid_base_arch=vid_base_arch, 
            aud_base_arch=aud_base_arch,
            pretrained=pretrained,
            norm_feat=norm_feat,
            use_mlp=use_mlp,
            num_classes=num_classes,
            args=args
        )
    elif model_type == 'vid_text_gdt':
        print('Using Video-Text GDT')
        model = TextVid_GDT(
            vid_base_arch=vid_base_arch, 
            pretrained=pretrained, 
            norm_feat=norm_feat, 
            use_mlp=use_mlp,
            num_classes=num_classes, 
        )
    else:
        print('Using Audio-Visual GDT')
        model = GDT(
            vid_base_arch=vid_base_arch, 
            aud_base_arch=aud_base_arch,
            pretrained=pretrained, 
            norm_feat=norm_feat, 
            use_mlp=use_mlp,
            num_classes=num_classes, 
        )
    return model
