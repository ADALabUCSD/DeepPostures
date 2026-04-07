from timm.models.vision_transformer import Block, Attention
import torch
import torch.nn as nn

from util.pos_embed import RotaryEmbedding2D, apply_rotary_pos_emb_2d


class AttentionWithRoPE(Attention):
    def __init__(self, embed_dim, num_heads,
                 max_time_len=256, max_chan_len=24,
                 base_time=500.0, base_chan=50.0,
                 qkv_bias=True, attn_drop=0.0, 
                 proj_drop=0.0, cls_token=True,
                 **attn_kwargs):
        super().__init__(dim=embed_dim, num_heads=num_heads,
                         qkv_bias=qkv_bias, attn_drop=attn_drop,
                         proj_drop=proj_drop, **attn_kwargs)
        head_dim = embed_dim // num_heads
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 so each axis half can rotate in pairs"
        self.rope2d = RotaryEmbedding2D(
            head_dim=head_dim,
            max_time_len=max_time_len,
            max_chan_len=max_chan_len,
            base_time=base_time,
            base_chan=base_chan,
            device=None,
        )
        self.cls_token = cls_token

    @staticmethod
    def _build_ids(chan_len: int, 
                   time_len: int, 
                   vis_index: torch.Tensor | None,
                   device: torch.device):
        C, T = chan_len, time_len
        time_full = torch.arange(T, device=device).repeat(C)             # [C*T]
        chan_full = torch.arange(C, device=device).repeat_interleave(T)  # [C*T]

        if vis_index is None:
            return time_full, chan_full

        if not torch.is_tensor(vis_index):
            raise TypeError("vis_index must be a tensor or None")

        if vis_index.dim() == 1:
            vis_index = vis_index.to(device=device, dtype=torch.long)    # cast here
            return time_full.index_select(0, vis_index), chan_full.index_select(0, vis_index)

        if vis_index.dim() == 2:
            vis_index = vis_index.to(device=device, dtype=torch.long)    # and here
            return time_full[vis_index], chan_full[vis_index]

        raise ValueError("vis_index must be None, 1D, or 2D")


    def forward(self, 
                x: torch.Tensor,
                vis_index: torch.Tensor|None,
                chan_len: int, 
                time_len: int):
        
        B, N, Cemb = x.shape

        if vis_index is None:
            nvis = chan_len * time_len
        else:
            nvis = vis_index.shape[-1]

        expected = nvis+1 if self.cls_token else nvis
        assert N == expected, f"N must equal cls_len plus visible tokens, got {N} vs {expected}"

        # project
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, Cemb // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, H, N, D]

        # split out CLS part
        if self.cls_token:
            q_cls, k_cls, v_cls = q[:, :, :1, :], k[:, :, :1, :], v[:, :, :1, :]
            q_tok, k_tok, v_tok = q[:, :, 1:, :], k[:, :, 1:, :], v[:, :, 1:, :]
        else:
            q_tok, k_tok, v_tok = q, k, v

        # build RoPE caches for patches only
        cos_t, sin_t, cos_c, sin_c = self.rope2d(
            time_len=time_len, chan_len=chan_len, device=x.device, dtype=x.dtype
        )
        time_ids, chan_ids = self._build_ids(chan_len, 
                                             time_len,  
                                             vis_index=vis_index,
                                             device=x.device)  # length C*T

        # rotate only the patch tokens
        q_tok, k_tok = apply_rotary_pos_emb_2d(
            q_tok, k_tok, cos_t, sin_t, cos_c, sin_c, time_ids, chan_ids
        )

        # stitch back
        if self.cls_token:
            q = torch.cat([q_cls, q_tok], dim=2)
            k = torch.cat([k_cls, k_tok], dim=2)
            v = torch.cat([v_cls, v_tok], dim=2)
        else:
            q, k, v = q_tok, k_tok, v_tok

        # attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, Cemb)
        out = self.proj_drop(self.proj(out))
        return out


class BlockWithRoPE(Block):
    """
    Swap timm Block attention with AttentionWithRoPE.
    """
    def __init__(self,
                 num_time_token: int, # 20
                 num_chan_token: int, # 3
                 embed_dim: int,
                 num_heads: int,
                 max_time_len: int = 256,
                 max_chan_len: int = 24,
                 base_time: float = 500.0,
                 base_chan: float = 50.0,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 init_values=None,
                 drop_path: float = 0.1,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 cls_token=True):
        # pass only what Block supports
        super().__init__(dim=embed_dim,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias,
                         init_values=init_values,
                         drop_path=drop_path,
                         act_layer=act_layer,
                         norm_layer=norm_layer)

        # replace attention, forwarding bias and drop probs to Attention
        self.attn = AttentionWithRoPE(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_time_len=max_time_len,
            max_chan_len=max_chan_len,
            base_time=base_time,
            base_chan=base_chan,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            cls_token=cls_token,
        )
        self.time_len = num_time_token
        self.chan_len = num_chan_token
        
    def forward(self, 
                x: torch.Tensor,
                vis_idx=None):
        x = x + self.drop_path1(self.attn(self.norm1(x), 
                                          chan_len=self.chan_len, time_len=self.time_len, vis_index=vis_idx))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

##
if __name__ == "__main__":
    torch.manual_seed(0)
    B, Cemb, num_heads = 2, 128, 8
    chan_len, time_len = 3, 20
    N = chan_len * time_len

    block = BlockWithRoPE(
        num_time_token=time_len,
        num_chan_token=chan_len,
        embed_dim=Cemb,
        num_heads=num_heads,
        max_time_len=500,
        max_chan_len=20,
        base_time=500.0,
        base_chan=20.0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        attn_drop=0.0,
        proj_drop=0.0,
        cls_token=False,
    )

    x = torch.randn(B, N, Cemb, requires_grad=True)
    y = block(x,)
    print("input:", x.shape, "output:", y.shape)
    y.pow(2).mean().backward()
    print("grad:", x.grad.shape)



