import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
#from util.pos_embed import tAPE
from einops import rearrange

class PatchEmbed_ts(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, ts_len=200, 
                 patch_size=20, 
                 embed_dim=512,
                 nvar=6, 
                 stride=20, ): # non-overlapping patches
        super().__init__()

        '''

        Input: raw_series (bs x nvar x L) 'Differ from Howon's construction of (bs x nvar x 1 x L)'
        
        bs x nvar x L -> bs x nvar x num_patches x patch_size -> bs x num_patches x (nvar*patch_size) -> bs x num_patches x E
        '''
        
        self.ts_len = ts_len
        self.patch_size = patch_size
        self.nvar = nvar
        self.num_patches = int(ts_len//patch_size)

        self.proj = nn.Linear(patch_size*nvar,embed_dim)


    def forward(self, x):
        # x: bs x nvar x L
        # Check dimensions consistency

        bs, nvar, L = x.shape
        assert L == self.num_patches * self.patch_size, "L must be equal to num_patches * patch_size"
        
        x = x.view(bs, nvar, self.num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).contiguous() # bs x num_patches, nvar, patch_size
        x = x.view(bs, self.num_patches, nvar*self.patch_size) # bs x num_patch x nvar* num_patch_size

        x = self.proj(x) # bs x num_patch x E

        return x

from transformers.activations import ACT2FN # hugging face api map string to activiation class. i.e. ACT2FN["gelu"]
import torch.nn.functional as F
class SundialPatchEmbedding(nn.Module):
    # develop feasible patch tokenization for arbitrary-length input time series
    # default is Sundial config
    '''
    fixed number of patches, but make patch size flexible.
    '''
    def __init__(self,
                 hidden_size=768,
                 intermediate_size=3072,
                 dropout_rate=0.1,
                 patch_size=10, # iwatch config
                 hidden_act='silu'):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_layer = nn.Linear(
            patch_size * 2, intermediate_size) # *2 because need to input the mask too.
        self.act = ACT2FN[hidden_act]
        self.output_layer = nn.Linear(
            intermediate_size, hidden_size)
        self.residual_layer = nn.Linear(
            patch_size * 2, hidden_size)
        self.patch_size = patch_size
    def forward(self, x):
        '''
        x: input tensor of shape [batch_size, nvar, seq_len]
        output: tensor of shape [batch_size, nvar, hidden_size]

        '''

        B, _, C, L = x.shape
        x = rearrange(x, 'b 1 c l -> (b c) l') 

        mask = torch.ones_like(x, dtype=torch.float32)
        input_length = x.shape[-1]
        padding_length = (self.patch_size - (input_length %
                          self.patch_size)) % self.patch_size
        x = F.pad(x, (padding_length, 0))
        mask = F.pad(mask, (padding_length, 0))
        # patchify x and mask
        x = x.unfold(dimension=-1, size=self.patch_size,
                     step=self.patch_size) # (b*c, num_patches, patch_size)
        mask = mask.unfold(dimension=-1, size=self.patch_size,
                           step=self.patch_size)

        x = torch.cat([x, mask], dim=-1)
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)
        out = out + res

        out = rearrange(out, '(b c) p e -> b (c p) e', b=B, c=C)

        return out
    
    """
    useage:
    self.embed_layer = SundialPatchEmbedding(config)
    input_ids is the input of time series, its shape is [batch_size, seq_len]
    inputs_embeds = self.embed_layer(input_ids)
    seq_length = inputs_embeds.shape[1]
    
    """


if __name__ == '__main__':
    # patch_emb = PatchEmbed_new(img_size=(387,65), patch_size=(9,5), in_chans=3, embed_dim=64, stride=(9,5))
    # input = torch.rand(8,3,387,65)
    # output = patch_emb(input)
    # print(output.shape) # (8,559,64)

    # patch_emb = PatchEmbed3D_new(video_size=(6,224,224), patch_size=(2,16,16), in_chans=3, embed_dim=768, stride=(2,16,16))
    # input = torch.rand(8,3,6,224,224)
    # output = patch_emb(input)
    #print(output.shape) # (8,64)

    # patch_emb = PatchEmbed_ts(ts_len=387,patch_size=9,stride=9)
    # input = torch.randn(6,387)
    # output = patch_emb(input)
    # print(output.shape)
    # print(patch_emb.patch_size)

    patch_embed = SundialPatchEmbedding(patch_size=10)
    input = torch.randn(6,1,3,300)
    output = patch_embed(input) # 6,90, 768
    print(output.shape)  
