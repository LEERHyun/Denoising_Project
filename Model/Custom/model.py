import torch
import torch.nn as nn
import torch.nn.functional as F
from arch_util import LayerNorm2d
from local_arch import Local_Base
import numbers
from einops import rearrange
import torchsummary

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Restormer Module
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
    
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

## TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
    
##########################################################################
## Resizing modules

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    
##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x
    

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#NAFNet Module
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#SimpleGate Module
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
#NAFBlock Architecture
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True) #
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


#----------------------------------------------------------------------------------------
#HybridModel
#----------------------------------------------------------------------------------------
class HybridNAFNet(nn.Module):

    def __init__(self, 
                 img_channel=3,
                 out_channel = 3,
                   width=32, 
                   middle_blk_num=6, 
                   enc_blk_nums=[2,1,3], 
                   dec_blk_nums=[2,1,1], 
                   refinement=2,
                   ffn_expansion_factor =2.66,
                   bias = False,
                   LayerNorm_type = "WithBias",
                   heads = [1,2,4,8]
                   ):
        super().__init__()


        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)# 3 =>32
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True) # 32=>3

        #Level 1 Encoder
        chan = width
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=chan, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(enc_blk_nums[0])])
        self.down1_2 = nn.Conv2d(chan, 2*chan, 2, 2) ## DownSample Level 1 to Level 2
        
        chan = chan*2
        
        #Level 2 Encoder
        self.encoder_level2 = nn.Sequential(*[NAFBlock(chan) for _ in range(enc_blk_nums[1])])
        self.down2_3 = nn.Conv2d(chan, 2*chan, 2, 2) ## From Level 2 to Level 3
        
        chan = chan*2
        
        #Level 3 Encoder
        self.encoder_level3 = nn.Sequential(*[NAFBlock(chan) for _ in range(enc_blk_nums[2])])
        self.down3_4 = nn.Conv2d(chan, 2*chan, 2, 2) ## From Level 3 to Level 4
        
        chan = chan*2
        
        #Middle Block
        self.middle = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        
        #Level 3 Decoder        
        self.up4_3 = nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                ) ## From Level 4 to Level 3
        chan = chan//2
        
        self.decoder_level3 = nn.Sequential(*[NAFBlock(chan) for _ in range(dec_blk_nums[2])])

        #Level 2 Decoder
        
        self.up3_2 = nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                ) ## From Level 4 to Level 3
        
        chan = chan//2        
        
        self.decoder_level2 = nn.Sequential(*[NAFBlock(chan) for _ in range(dec_blk_nums[1])])
        
        #Level 1 Decoder
        self.up2_1 = nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                ) #  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        chan = chan//2
        
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=chan, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(dec_blk_nums[0])])
        
        #Refinement
        self.refinement = nn.Sequential(*[NAFBlock(chan) for _ in range(refinement)])
        
        #Ending
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True) # 32=>3    
        
        self.padder_size = 2 ** (len(enc_blk_nums)+1)


    def forward(self, inp):
        
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)        
        inp_enc_level1 = self.intro(inp) # 32->3
        
        #Level 1 Encoder
        out_enc_level1 = self.encoder_level1(inp_enc_level1) #Skip Connection Value Level 1
        inp_enc_level2 = self.down1_2(out_enc_level1) #Dowsample 1->2
        
        #Level 2 Encoder
        out_enc_level2 = self.encoder_level2(inp_enc_level2) #Skip Connection Value Level 2
        inp_enc_level3 = self.down2_3(out_enc_level2) #Downsample 2->3
        
        #Level 3 Encoder
        out_enc_level3 = self.encoder_level3(inp_enc_level3) #Skip Connection Value Level 3
        inp_enc_level4 = self.down3_4(out_enc_level3) #Downsample 3->4
        
        #Middle Block
        latent = self.middle(inp_enc_level4) 
        
        #Level 3 Decoder
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = inp_dec_level3 + out_enc_level3 #Skip connection Level 3
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        #Level 2 Decoder
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = inp_dec_level2 + out_enc_level2 #Skip connection Level 2
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        #Level 2 Decoder
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = inp_dec_level1 + out_enc_level1 #Skip connection Level1
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        #Refinement
        refinement_out = self.refinement(out_dec_level1)
        
        #32->3
        ending_out = self.ending(refinement_out)
        
        output = ending_out + inp  #Skip Connection Input+Output              
        return output[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x



if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    img_channel = 3
    width = 32



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inp_shape = (3,256,256)




    #Calculate Model Complexity---------------------------------------------------------------------------------------------------------------------------------------------------------
    
    custom = HybridNAFNet(enc_blk_nums=[4,2,2],middle_blk_num=16,dec_blk_nums=[4,2,2],refinement=4)
    custom.to(device)
    #Model Summary
    torchsummary.summary(custom,inp_shape)
    flops, params = get_model_complexity_info(custom, (3,256,256), verbose=False, print_per_layer_stat=False)
    #torchsummary.summary(res, inp_shape)   
    print('{:<30}  {:<8}'.format('Computational Complexity: ', flops))
    print('{:<30}  {:<8}'.format('Computational Complexity: ', params))
