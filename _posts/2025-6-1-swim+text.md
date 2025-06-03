下面给出一个较为完整的思路与示例代码片段，演示如何把 Swin‐Transformer（即你代码中的 CRSTB/Cross_WindowAttention 结构）与“文本指示”融合起来，使得网络在做图像窗口注意力时，也能“看到”一句话或者一段描述，并据此调整特征融合。整体分为四步：

1. **准备文本输入**：利用一个现成的文本编码器（如 CLIP Text Encoder）把文字转成一组 embedding。
2. **投影到与图像同维度**：把文本 embedding 线性映射到与 Swin 特征维度相同的维度，以便后续拼接。
3. **在 Cross_WindowAttention 中拼接图像 KV 与文本 KV**：借助前面讲过的“把文本 KV 复制到每个窗口，再 concat 到图像 KV 上”思路，让每个图像 Query 都能 attend 文本。
4. **在 CRSTB 中调用带文本的跨模态注意力**：在 forward_features_Fusion（或对应 Fusion 层）里，把文本 embedding 先拿出来，然后传给每个 Cross_SwinTransformerBlock。

下面假设你已经有了原先那段 SwinFusion 代码，我们只需在 Fusion 部分插入“文本制导”即可。为了说清楚，我会重点突出关键改动，其他部分保持原样（省略不变代码）。其中所有新加／替换的行都已经打上了“## ←”注释：

------

## 1. 安装并加载 CLIP 文本编码器

首先，需要安装并导入 CLIP。这里以 HuggingFace 上的 `CLIPTextModel` 为例（你也可以用 OpenAI 官方的 CLIP 库，原理是一样的）：

```bash
pip install transformers
from transformers import CLIPTokenizer, CLIPTextModel

# 在 SwinFusion.__init__ 或者最上面 import 区域添加：
self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")     ## ←
self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32") ## ←
# 冻结文本编码器权重（如果你只想用预训练，不做微调的话）：
for p in self.text_encoder.parameters():
    p.requires_grad = False
# 文本投影到 Swin 特征维度（假设 embed_dim = self.embed_dim）
self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, self.embed_dim) ## ←
```

- `CLIPTokenizer`：把一句话 tokenization 成 `input_ids, attention_mask`。
- `CLIPTextModel`：把 tokens 编码成 `(B, L_t, 512)`（512 是 CLIP 默认的隐藏维）。
- `self.text_proj`：把 512 维的 CLIP 文本特征投影成 `embed_dim` 维，以便与 Swin 图像特征在同一维度拼接。

------

## 2. 修改 Cross_WindowAttention，使其支持“额外的文本 KV 拼接”

我们先把原来的 `Cross_WindowAttention` 改成：

```python
class Cross_WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 图像–图像的相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # Query 来自 x，Key/Value 来自 y 或文本
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, t_feats=None, mask=None):
        """
        x: (nW*B, N_img, C)          ← Query 来自某模态窗口
        y: (nW*B, N_img, C)          ← Key/Value 来自另一模态窗口
        t_feats: None 或 (B, L_t, C) ← 文本特征投影到 C 维
        mask:  None 或注意力掩码 (nW, N_img, N_img)
        """
        Bn, N_img, C = x.shape
        nH = self.num_heads
        C_head = C // nH

        # 1) Q 线性映射并拆多头
        q = self.q(x).reshape(Bn, N_img, nH, C_head).permute(0, 2, 1, 3)  # (Bn, nH, N_img, C_head)

        # 2) “纯图像”的 K,V
        kv_img = self.kv(y)  # → (Bn, N_img, 2*C)
        kv_img = kv_img.view(Bn, N_img, 2, nH, C_head).permute(2, 0, 3, 1, 4)
        k_img, v_img = kv_img[0], kv_img[1]  # (Bn, nH, N_img, C_head) ×2

        if t_feats is not None:
            # t_feats: (B, L_t, C) 
            B = t_feats.shape[0]
            # 计算 nW 从 Bn、B、N_img 得到： nW*B = Bn → nW = Bn // B
            nW = Bn // B
            L_t = t_feats.shape[1]

            # a) 复制文本到所有窗口
            #    t_feats.unsqueeze(1): (B, 1, L_t, C) → repeat 到 (B, nW, L_t, C)
            t_rep = t_feats.unsqueeze(1).repeat(1, nW, 1, 1)  # (B, nW, L_t, C)
            t_rep = t_rep.view(Bn, L_t, C)                    # (Bn, L_t, C)

            # b) 通过同一个 self.kv 投影，得到文本 KV
            kv_txt = self.kv(t_rep)  # (Bn, L_t, 2*C)
            kv_txt = kv_txt.view(Bn, L_t, 2, nH, C_head).permute(2, 0, 3, 1, 4)
            k_txt, v_txt = kv_txt[0], kv_txt[1]  # (Bn, nH, L_t, C_head) ×2

            # c) 拼接图像 KV 与 文本 KV
            k = torch.cat([k_img, k_txt], dim=2)  # (Bn, nH, N_img + L_t, C_head)
            v = torch.cat([v_img, v_txt], dim=2)  # (Bn, nH, N_img + L_t, C_head)
            N_total = N_img + L_t
        else:
            k, v = k_img, v_img
            N_total = N_img

        # 3) 计算注意力分数
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (Bn, nH, N_img, N_total)

        # 4) 加上相对位置偏置（只对图像–图像子矩阵有效）
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N_img, N_img, -1)             # (N_img, N_img, nH)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # (1, nH, N_img, N_img)

        if t_feats is not None:
            # 把 (1,nH,N_img,N_img) 嵌入到 (1,nH,N_img,N_total) 的左上角
            bias = torch.zeros((1, nH, N_img, N_total), device=attn.device)
            bias[:, :, :N_img, :N_img] = relative_position_bias
            attn = attn + bias
        else:
            attn = attn + relative_position_bias

        # 5) 如果有 mask，需要扩展到 N_total 维
        if mask is not None:
            nW_local = mask.shape[0]
            # 原本 mask: (nW, N_img, N_img) → (B, nW, nH, N_img, N_img)
            # 需要扩到 (B, nW, nH, N_img, N_total)，对额外的 N_total-N_img 列不做屏蔽
            attn = attn.view(Bn // nW_local, nW_local, nH, N_img, N_total)
            mask2 = mask.unsqueeze(1).unsqueeze(-1).repeat(1, nH, 1, N_total)  # (nW, nH, N_img, N_total)
            mask2 = mask2.unsqueeze(0).repeat(Bn//nW_local, 1, 1, 1, 1)        # (B, nW, nH, N_img, N_total)
            attn = attn + mask2
            attn = attn.view(Bn, nH, N_img, N_total)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # 6) 加权求和
        out = (attn @ v)  # (Bn, nH, N_img, C_head)
        out = out.permute(0, 2, 1, 3).reshape(Bn, N_img, C)  # (Bn, N_img, C)

        # 7) 输出线性映射
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def extra_repr(self):
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"
```

**关键点说明：**

- `forward(self, x, y, t_feats=None, mask=None)`
  - `x`: 当前模态的窗口序列（如来自 A 分支）。
  - `y`: 另一个模态的窗口序列（如来自 B 分支）。
  - `t_feats`：形状 (B,Lt,C)(B, L_t, C)，是“投影后”的文本 token 特征，如果不需要文本就传 `None`。
  - 最后如果 `t_feats` 不为 `None`，先把它复制到每个窗口，对所有窗口做同样的 KV 投影，然后与 `k_img, v_img` 拼接。
- **拼接后 Attention**：原来 `q` 形状 (Bn,nH,Nimg,Chead)(Bn, nH, N_img, C_head)，`k_img` 形状 (Bn,nH,Nimg,Chead)(Bn, nH, N_img, C_head)，拼接成 (Bn,nH,Nimg+Lt,Chead)(Bn, nH, N_img + L_t, C_head)。最终 `attn` 形状 (Bn,nH,Nimg,Nimg+Lt)(Bn, nH, N_img, N_img + L_t)，也就是每个图像 token 同时 attend 到其他图像 token 及全部文本 token。
- **相对位置偏置**：我们只沿用原来那张 (Nimg,Nimg)(N_img, N_img) 偏置矩阵，把它放到扩展后的左上角，其余和“图像–文本”对应的位置全部填 0。

------

## 3. 在 CRSTB 里调用带文本的跨模态注意力

接着把 `Cross_SwinTransformerBlock` 的 `forward` 改写一下，让它能接收 `t_feats`：

```python
class Cross_SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_text=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # 如果窗口过大，就不用移位
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.use_text = use_text  # ← 是否要用文本
        # 如果 use_text=True，需要提前在上层把 t_feats 准备好

        # 两个分支各自的 LayerNorm
        self.norm1_A = norm_layer(dim)
        self.norm1_B = norm_layer(dim)

        # 跨模态注意力：A_query→B_KV；B_query→A_KV
        self.attn_A = Cross_WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_B = Cross_WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path_A = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_B = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_A = norm_layer(dim)
        self.norm2_B = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_A = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.mlp_B = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        if self.shift_size > 0:
            self.attn_mask = self.calculate_mask(self.input_resolution)
        else:
            self.attn_mask = None

    def calculate_mask(self, x_size):
        # 与之前完全相同，不赘述
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x, y, x_size, t_feats=None):
        """
        x: (B, H*W, C)
        y: (B, H*W, C)
        x_size: (H, W)
        t_feats: None 或 (B, L_t, C)  ← 投影后文本特征
        """
        B, L, C = x.shape
        H, W = x_size
        shortcut_A, shortcut_B = x, y

        # 1. 归一化 + reshape 成 (B, H, W, C)
        x = self.norm1_A(x).view(B, H, W, C)
        y = self.norm1_B(y).view(B, H, W, C)

        # 2. 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x, shifted_y = x, y

        # 3. 窗口划分
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, window_size, window_size, C)
        y_windows = window_partition(shifted_y, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (nW*B, N_img, C)
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)

        # 4. 跨模态 WindowAttention，传入 t_feats
        if self.attn_mask is not None:
            mask = self.attn_mask.to(x.device)
        else:
            mask = None

        if self.use_text and (t_feats is not None):
            # 直接在所有窗口里“拼接”同一份 t_feats
            attn_windows_A = self.attn_A(x_windows, y_windows, t_feats, mask=mask)  # (nW*B, N_img, C)
            attn_windows_B = self.attn_B(y_windows, x_windows, t_feats, mask=mask)
        else:
            attn_windows_A = self.attn_A(x_windows, y_windows, None, mask=mask)
            attn_windows_B = self.attn_B(y_windows, x_windows, None, mask=mask)

        # 5. 窗口还原
        attn_windows_A = attn_windows_A.view(-1, self.window_size, self.window_size, C)  # (nW*B,Wh,Ww,C)
        attn_windows_B = attn_windows_B.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows_A, self.window_size, H, W)  # (B, H, W, C)
        shifted_y = window_reverse(attn_windows_B, self.window_size, H, W)

        # 6. 逆向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            y = torch.roll(shifted_y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x, y = shifted_x, shifted_y

        x = x.view(B, H * W, C)
        y = y.view(B, H * W, C)

        # 7. 残差 + MLP
        x = shortcut_A + self.drop_path_A(x)
        x = x + self.drop_path_A(self.mlp_A(self.norm2_A(x)))

        y = shortcut_B + self.drop_path_B(y)
        y = y + self.drop_path_B(self.mlp_B(self.norm2_B(y)))

        return x, y

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, " \
               f"num_heads={self.num_heads}, window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
```

**要点总结：**

- 增加了 `use_text` 标志，表示这一层是否要调用带文本的注意力。
- 如果 `t_feats is not None`，就把文本传进去，Cross_WindowAttention 会自动把它拼到 KV 阶段。
- 如果后续不想某些 Fusion 层用文本，就可以在调用时传 `t_feats=None`。

------

## 4. 在 SwinFusion.forward_features_Fusion 里把文本 embedding 传进去

最后，要在主网络的 Fusion 阶段准备 `t_feats`，并把它传给每一层 CRSTB。修改 `forward_features_Fusion` 如下（核心部分）：

```python
def forward_features_Fusion(self, x, y, text_list=None):
    """
    x, y: (B, C, H, W)  来自上游 Ex_A/Ex_B 分支的特征
    text_list: None 或 ["一句话描述", ...]  长度 == B
    """

    B, C, H, W = x.shape
    x_size = (H, W)

    # 1. PatchEmbed → (B, H*W, C)
    x_seq = self.patch_embed(x)
    y_seq = self.patch_embed(y)
    if self.ape:
        x_seq = x_seq + self.absolute_pos_embed
        y_seq = y_seq + self.absolute_pos_embed
    x_seq = self.pos_drop(x_seq)
    y_seq = self.pos_drop(y_seq)

    # 2. 如果传了 text_list，就先做文本编码，得到 t_feats (B, L_t, C)
    if text_list is not None and self.cfg.use_text:
        # text_list: List[str]，长度 B
        tokens = self.tokenizer(
            text_list, padding='longest',
            truncation=True, max_length=77,
            return_tensors='pt'
        ).to(x.device)  # (B, L_t)
        txt_outputs = self.text_encoder(**tokens)         # CLIPTextModel 输出
        txt_embeds = txt_outputs.last_hidden_state       # (B, L_t, clip_dim)
        t_feats = self.text_proj(txt_embeds)             # (B, L_t, C)
    else:
        t_feats = None

    # 3. 逐层 Fusion：把 t_feats 传给 Cross_SwinTransformerBlock
    for i, layer in enumerate(self.layers_Fusion):
        # 这里举例：只有第一层融合文本，后面层不再用文本
        if self.cfg.use_text and (i == 0) and (t_feats is not None):
            x_seq, y_seq = layer(x_seq, y_seq, x_size, t_feats)
        else:
            x_seq, y_seq = layer(x_seq, y_seq, x_size, None)

    # 4. Norm + UnEmbed 回 (B,C,H,W)
    x_out = self.norm_Fusion_A(x_seq)
    x_out = self.patch_unembed(x_out, x_size)  # (B, C, H, W)
    y_out = self.norm_Fusion_B(y_seq)
    y_out = self.patch_unembed(y_out, x_size)

    # 5. 拼通道 + Conv 下采样
    fused = torch.cat([x_out, y_out], dim=1)      # (B, 2C, H, W)
    fused = self.lrelu(self.conv_after_body_Fusion(fused))  # (B, C, H, W)
    return fused
```

**要点说明：**

- `text_list`：外面再调用 `forward` 时，把一组 batch 大小相同的字符串传进来。例如 `["这是一次夜景增强", "请突出主体色彩", ...]`，长度等于当前 Batch 大小 `B`。
- 利用 CLIP 的 `tokenizer`+`text_encoder`，得到 `(B, L_t, clip_dim)`，再线性映射到 `(B, L_t, C)`。
- 在循环 Fusion 层时，只有 `i==0` 的第一层把 `t_feats` 传下去；后续层按需传 `None`。你可以根据实际需求改成“每层都传”或者“第 0、2 层传”。

------

## 5. 在 SwinFusion.forward 里传入 `text_list`

最后，把主网络的 `forward` 修改成：

```python
def forward(self, A, B, text_list=None):
    """
    A, B: (B, 1, H, W)  输入的两张单通道图像（或多通道）
    text_list: None 或 List[str]，长度 == B
    """
    x, y = A, B
    H, W = x.shape[2:]
    x = self.check_image_size(x)
    y = self.check_image_size(y)

    self.mean_A = self.mean.type_as(x)
    self.mean_B = self.mean.type_as(y)
    self.mean = (self.mean_A + self.mean_B) / 2
    x = (x - self.mean_A) * self.img_range
    y = (y - self.mean_B) * self.img_range

    # 1. Ex_A / Ex_B
    deep_A = self.forward_features_Ex_A(x)  # (B, C, H, W)
    deep_B = self.forward_features_Ex_B(y)  # (B, C, H, W)

    # 2. Fusion (加入文本控制)
    fused = self.forward_features_Fusion(deep_A, deep_B, text_list)  # (B, C, H, W)

    # 3. Reconstruction
    out = self.forward_features_Re(fused)  # (B, C_out, H*up, W*up)

    out = out / self.img_range + self.mean
    return out[:, :, :H*self.upscale, :W*self.upscale]
```

这样，只要在调用主网络时传入 `text_list`，就会在 Fusion 的第一层让所有窗口都“看到”这一句或几句话，并据此调整跨模态注意力。

------

## 6. 小结

1. **文本编码**：先用 CLIP Text Encoder（或任意你喜欢的 Text Model）把句子映射为 `(B, L_t, text_dim)`。
2. **投影对齐**：用一层 `nn.Linear(text_dim→embed_dim)`，把文本特征映射到与 Swin 图像特征一致的维度。
3. **Cross_WindowAttention 拼接**：原理同前面多次讲过的“拼接图像 KV 与文本 KV”，即在 `Cross_WindowAttention.forward` 中：
   - 先做图像 KV、
   - 如果 `t_feats` 不为空，就把 `t_feats` 复制到每个窗口 → 走同一个 `kv` 线性层 → 拆头 → 得到 `(Bn, nH, L_t, C_head)` 的 `k_txt, v_txt`。
   - 再与 `(Bn, nH, N_img, C_head)` 的 `k_img, v_img` 在“序列长度”维度上 concat → 形成 `(Bn, nH, N_img+L_t, C_head)`，接着照常做 Attention。
4. **CRSTB 调用**：在循环 Fusion 层时，如果要用文本，就传 `t_feats`；否则传 `None`。所以你可以灵活选择“只在第 1 层融合文本”或“跨多个层都融合”或“仅在最后一层融合”。
5. **训练细节**：
   - 如果你仅用 CLIP Text Encoder 做前向冻结，就不要给它设优化器。
   - SwinTransformer 部分参数和 CLIP Text 投影部分参数一起训练或微调即可。
   - 由于把文本复制到每个窗口会带来计算与内存开销，若分辨率很大，建议把 `use_text` 只打开最浅的 1–2 层 Fusion，或者先把文本做“关键词抽取”后再传入，减少 `L_t`。
6. **可选优化**：
   - 如果 `L_t` 足够短，可以直接传整句；若 `L_t` 很长，可以先只保留前 16–32 个 token，或者做 CLIP 文本池化（取最后一个 `[EOS]` token）→ 得到 `(B, C)` → 再复制到每个窗口复制成 `(B, 1, C)` → 拼接到图像 KV→ 性能更高，但语义稍弱。
   - 可以在 `Cross_WindowAttention` 里为 “图像–文本” 区域加一个小的比例因子或可学习偏置，让模型更习惯给图像–文本注意力一个不同权重。

这样一来，你就实现了“让 Swin‐Transformer 的每个窗口不仅能 attend 另一个模态的图像，还能 attend 来自文本提示的 token”的融合方式。最终网络就能在做图像超分（或其他任务）时，依据传入的文字指示（如“降低噪点”、“增强对比度”）来动态调整特征融合。

希望这个思路能帮到你！如果后面还需要更深层的优化（比如在多层都插入文本、或者让文本也在图像自身的 RSTB 里做 FiLM 调制），可以在此基础上进一步扩展。