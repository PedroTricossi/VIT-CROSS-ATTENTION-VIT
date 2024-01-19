import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ViT
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def default(val, d):
    return val if val is not None else d

# ViT & CrossViT
# PreNorm class for layer normalization before passing the input to another function (fn)
class PreNorm(nn.Module):    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# ViT & CrossViT
# FeedForward class for a feed forward neural network consisting of 2 linear layers, 
# where the first has a GELU activation followed by dropout, then the next linear layer
# followed again by dropout
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

# ViT & CrossViT
# Attention class for multi-head self-attention mechanism with softmax and dropout

# For the purpose of the exercise, the attention model is based in UvA attention implementatoin
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.query_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.key_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.value_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads

        x = self.norm(x)
        context = default(context, x)

        # Project query, key, and value vectors.
        query = self.query_proj(x)
        key = self.key_proj(context)
        value = self.value_proj(context)

        # Split query, key, and value vectors into heads.
        # (b, n, h, dim_head)
        # (b, h, n, dim_head)

        query = query.view(-1,  self.heads, query.shape[1], self.dim_head)
        key = key.view(-1, self.heads, key.shape[1], self.dim_head)
        value = value.view(-1,  self.heads, value.shape[1], self.dim_head)

        attention_scores = einsum('b h i d, b h j d -> b h i j', query, key) * self.scale

        attention_weights = torch.softmax(attention_scores, dim=-1)

        attention_output = torch.einsum('bhij, bhjd -> bhid', attention_weights, value)

        attention_output = attention_output.view(-1, attention_output.shape[1], self.heads * self.dim_head)
        out = self.out_proj(attention_output)

        # Apply dropout to output.
        out = self.dropout(out)

        return out


# ViT & CrossViT
# CrossViT performs PreNorm -> Attention, but as this is the code given I didn't change it
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ])
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# CrossViT
# projecting CLS tokens, if the dimensions don't match, use a linear layer to project it in the right dimension
class ProjectInOut(nn.Module):
    """
    Adapter class that embeds a callable (layer) and handles mismatching dimensions
    """
    def __init__(self, dim_outer, dim_inner, fn):
        """
        Args:
            dim_outer (int): Input (and output) dimension.
            dim_inner (int): Intermediate dimension (expected by fn).
            fn (callable): A callable object (like a layer).
        """
        super().__init__()
        self.fn = fn

        need_projection = dim_outer != dim_inner

        self.project_in = nn.Linear(dim_outer, dim_inner) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_inner, dim_outer) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        """
        Args:
            *args, **kwargs: to be passed on to fn

        Notes:
            - after calling fn, the tensor has to be projected back into it's original shape   
            - fn(W_in) * W_out
        """
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs) + x
        x = self.project_out(x)
        return x

# CrossViT
# cross attention transformer
class CrossTransformer(nn.Module):
    # This is a special transformer block
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(sm_dim, lg_dim, PreNorm(lg_dim, Attention(lg_dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                ProjectInOut(lg_dim, sm_dim, PreNorm(sm_dim, Attention(sm_dim, heads = heads, dim_head = dim_head, dropout = dropout))),
            ]))


    def forward(self, sm_tokens, lg_tokens):
        sm_cls, sm_patch_tokens = sm_tokens[:, :1], sm_tokens[:, 1:]
        lg_cls, lg_patch_tokens = lg_tokens[:, :1], lg_tokens[:, 1:]
        
        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context = lg_patch_tokens, kv_include_self = True) 
            lg_cls = lg_attend_sm(lg_cls, context = sm_patch_tokens, kv_include_self = True)

        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim=1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim=1)

        return sm_tokens, lg_tokens

# CrossViT
# multi-scale encoder
class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(sm_dim, **sm_enc_params),
                Transformer(lg_dim, **lg_enc_params),
                CrossTransformer(sm_dim, lg_dim, cross_attn_depth, cross_attn_heads, cross_attn_dim_head, dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens = sm_enc(sm_tokens)
            lg_tokens = lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)



        return sm_tokens, lg_tokens

# CrossViT (could actually also be used in ViT)
# helper function that makes the embedding from patches
# have a look at the image embedding in ViT
class ImageEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        dropout = 0.
    ):
        super().__init__()


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)


    def forward(self, img):
        x = self.to_patch_embedding(img)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        return self.dropout(x)


# normal ViT
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # initialize patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # create transformer blocks
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # create mlp head (layer norm + linear layer)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # concat class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        # add positional embedding

        x += self.pos_embedding[:, :(n + 1)]

        # apply dropout
        x = self.dropout(x)

        # forward through the transformer blocks
        x = self.transformer(x)

        # decide if x is the mean of the embedding 
        # or the class token
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # transfer via an identity layer the cls tokens or the mean
        # to a latent space, which can then be used as input
        # to the mlp head
        x = self.to_latent(x)
        
        return self.mlp_head(x)


# CrossViT
class CrossViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        sm_dim,
        lg_dim,
        sm_patch_size = 12,
        sm_enc_depth = 1,
        sm_enc_heads = 8,
        sm_enc_mlp_dim = 2048,
        sm_enc_dim_head = 64,
        lg_patch_size = 16,
        lg_enc_depth = 4,
        lg_enc_heads = 8,
        lg_enc_mlp_dim = 2048,
        lg_enc_dim_head = 64,
        cross_attn_depth = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        depth = 3,
        dropout = 0.1,
        emb_dropout = 0.1
    ):
        super().__init__()
        # create ImageEmbedder for small and large patches
        self.sm_image_embedder = ImageEmbedder(dim = sm_dim, image_size = image_size, patch_size = sm_patch_size, dropout = emb_dropout)
        self.lg_image_embedder = ImageEmbedder(dim = lg_dim, image_size = image_size, patch_size = lg_patch_size, dropout = emb_dropout)


        # create MultiScaleEncoder
        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

        # create mlp heads for small and large patches
        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

    def forward(self, img):
        sm_tokens = self.sm_image_embedder(img)
        lg_tokens = self.lg_image_embedder(img)

        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))

        sm_logits = self.sm_mlp_head(sm_cls)

        lg_logits = self.lg_mlp_head(lg_cls)
        
        return sm_logits + lg_logits


if __name__ == "__main__":
    x = torch.randn(16, 3, 32, 32)
    vit = ViT(image_size = 32, patch_size = 8, num_classes = 10, dim = 64, depth = 2, heads = 8, mlp_dim = 128, dropout = 0.1, emb_dropout = 0.1)
    cvit = CrossViT(image_size = 32, num_classes = 10, sm_dim = 64, lg_dim = 128, sm_patch_size = 8,
                    sm_enc_depth = 2, sm_enc_heads = 8, sm_enc_mlp_dim = 128, sm_enc_dim_head = 64,
                    lg_patch_size = 16, lg_enc_depth = 2, lg_enc_heads = 8, lg_enc_mlp_dim = 128,
                    lg_enc_dim_head = 64, cross_attn_depth = 2, cross_attn_heads = 8, cross_attn_dim_head = 64,
                    depth = 3, dropout = 0.1, emb_dropout = 0.1)
    print(vit(x).shape)
    print(cvit(x).shape)
