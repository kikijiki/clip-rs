use std::path::Path;

use candle_core::{DType, Device, IndexOp, Module, ModuleT, Result, Tensor, Var, D};
use candle_nn::{
    batch_norm, conv2d, conv2d_no_bias, linear, linear_no_bias, ops::softmax_last_dim, BatchNorm,
    BatchNormConfig, Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear, VarBuilder, VarMap,
};
use tracing::info;

// https://github.com/openai/CLIP/blob/main/clip/model.py

const EXPANSION: usize = 4;

pub fn norm(t: &Tensor) -> Result<Tensor> {
    let norm = t.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    t.broadcast_div(&norm)
}

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Relu,
    QuickGelu,
    Gelu,
    GeluErf,
}

impl Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Activation::Relu => xs.relu(),
            Activation::QuickGelu => xs * candle_nn::ops::sigmoid(&(xs * 1.702f64)?)?,
            Activation::Gelu => xs.gelu(),
            Activation::GeluErf => xs.gelu_erf(),
        }
    }
}

pub struct Sequence {
    layers: Vec<Box<dyn Module>>,
}

impl Sequence {
    fn new() -> Self {
        Self { layers: Vec::new() }
    }

    fn add(&mut self, layer: impl Module + 'static) {
        self.layers.push(Box::new(layer));
    }
}

impl Module for Sequence {
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let mut x = image.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}

#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
}

impl MultiHeadAttention {
    fn new(
        n_state: usize,
        output_dim: Option<usize>,
        n_head: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        info!("Load {}", vb.prefix());

        let query;
        let key;
        let value;

        if vb.contains_tensor("in_proj_weight") {
            let in_proj_weight = vb.get((n_state * 3, n_state), "in_proj_weight")?;
            let in_proj_bias = vb.get((n_state * 3,), "in_proj_bias")?;

            query = Linear::new(
                in_proj_weight.i((0..n_state, ..))?,
                Some(in_proj_bias.i(0..n_state)?),
            );
            key = Linear::new(
                in_proj_weight.i((n_state..n_state * 2, ..))?,
                Some(in_proj_bias.i(n_state..n_state * 2)?),
            );
            value = Linear::new(
                in_proj_weight.i((n_state * 2..n_state * 3, ..))?,
                Some(in_proj_bias.i(n_state * 2..n_state * 3)?),
            );
        } else {
            query = linear(n_state, n_state, vb.pp("q_proj"))?;
            key = linear_no_bias(n_state, n_state, vb.pp("k_proj"))?;
            value = linear(n_state, n_state, vb.pp("v_proj"))?;
        }

        let out = linear(n_state, output_dim.unwrap_or(n_state), vb.pp("out_proj"))?;

        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (tgt_len, bsz, embed_dim) = x.dims3()?;

        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;

        let attn = self.qkv_attention(&q, &k, &v, mask)?;
        let out = self.out.forward(&attn)?;
        let out = out.reshape(&[tgt_len, bsz, embed_dim])?;

        Ok(out)
    }

    fn reshape_head(&self, x: &Tensor) -> Result<Tensor> {
        let (n_ctx, n_batch, n_state) = x.dims3()?;
        let x = x.reshape(&[n_ctx, n_batch * self.n_head, n_state / self.n_head])?;
        let x = x.transpose(0, 1)?;
        let x = x.reshape(&[n_batch, self.n_head, n_ctx, n_state / self.n_head])?;
        Ok(x)
    }

    pub fn scaled_dot_product_attention(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let device = query.device();
        let l = query.dim(D::Minus2)?;
        let s = key.dim(D::Minus2)?;
        let dim = query.dim(D::Minus1)?;

        let scale_factor = 1.0 / (dim as f64).sqrt();

        let mut attn_bias = Tensor::zeros((l, s), query.dtype(), device)?;

        if let Some(attn_mask) = attn_mask {
            if attn_mask.rank() > attn_bias.rank() {
                attn_bias = attn_bias.broadcast_as(attn_mask.shape())?;
            }
            attn_bias = (&attn_bias
                + attn_mask
                    .to_dtype(attn_bias.dtype())?
                    .broadcast_as(attn_bias.shape())?)?;
        }

        let mut attn_weights =
            (query.matmul(&key.transpose(D::Minus2, D::Minus1)?.contiguous()?)? * scale_factor)?;

        attn_weights = (&attn_weights + attn_bias.broadcast_as(attn_weights.shape())?)?;
        attn_weights = softmax_last_dim(&attn_weights)?;
        attn_weights.matmul(value)
    }

    fn qkv_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (tgt_len, bsz, embed_dim) = q.dims3()?;

        let q = self.reshape_head(q)?;
        let k = self.reshape_head(k)?;
        let v = self.reshape_head(v)?;

        let attn = Self::scaled_dot_product_attention(&q, &k, &v, mask)?;
        let attn = attn
            .permute((2, 0, 1, 3))?
            .contiguous()?
            .reshape(&[bsz * tgt_len, embed_dim])?;

        Ok(attn)
    }
}

pub struct Downsample {
    stride: usize,
    conv: Conv2d,
    bn: BatchNorm,
}

impl Downsample {
    fn new(in_planes: usize, planes: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        info!("Load {}", vb.prefix());

        let conv = conv2d(in_planes, planes, 1, Conv2dConfig::default(), vb.clone())?;
        let bn = batch_norm(planes, BatchNormConfig::default(), vb)?;
        Ok(Self { stride, conv, bn })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.avg_pool2d(self.stride)?;
        let x = self.conv.forward(&x)?;
        let x = self.bn.forward_t(&x, false)?;
        Ok(x)
    }
}

pub struct Bottleneck {
    stride: usize,
    conv1: Conv2d,
    bn1: BatchNorm,
    act1: Activation,
    conv2: Conv2d,
    bn2: BatchNorm,
    act2: Activation,
    conv3: Conv2d,
    bn3: BatchNorm,
    act3: Activation,
    downsample: Option<Downsample>,
}

impl Bottleneck {
    pub fn new(
        in_planes: usize,
        planes: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Bottleneck> {
        info!("Load {}", vb.prefix());

        let conv1 = conv2d(in_planes, planes, 1, Conv2dConfig::default(), vb.clone())?;
        let bn1 = batch_norm(planes, BatchNormConfig::default(), vb.clone())?;
        let act1 = Activation::Relu;

        let conv2 = conv2d(
            planes,
            planes,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.clone(),
        )?;
        let bn2 = batch_norm(planes, BatchNormConfig::default(), vb.clone())?;
        let act2 = Activation::Relu;

        let conv3 = conv2d(
            planes,
            planes * EXPANSION,
            1,
            Conv2dConfig::default(),
            vb.clone(),
        )?;
        let bn3 = batch_norm(planes * EXPANSION, BatchNormConfig::default(), vb.clone())?;
        let act3 = Activation::Relu;

        let downsample = if stride > 1 || in_planes != planes * EXPANSION {
            Some(Downsample::new(
                in_planes,
                planes * EXPANSION,
                stride,
                vb.clone(),
            )?)
        } else {
            None
        };

        Ok(Bottleneck {
            stride,
            conv1,
            bn1,
            act1,
            conv2,
            bn2,
            act2,
            conv3,
            bn3,
            act3,
            downsample,
        })
    }
}

impl Module for Bottleneck {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x.clone();

        let x = self.conv1.forward(&x)?;
        let x = self.bn1.forward_t(&x, false)?;
        let x = self.act1.forward(&x)?;

        let x = self.conv2.forward(&x)?;
        let x = self.bn2.forward_t(&x, false)?;
        let x = self.act2.forward(&x)?;

        let x = if self.stride > 0 {
            x.avg_pool2d(self.stride)?
        } else {
            x
        };

        let x = self.conv3.forward(&x)?;
        let x = self.bn3.forward_t(&x, false)?;

        let x = if let Some(ref downsample) = self.downsample {
            downsample.forward(&x)?
        } else {
            x
        };

        let out = x.add(&identity)?;
        let out = self.act3.forward(&out)?;

        Ok(out)
    }
}

pub struct AttentionPool2d {
    positional_embedding: Tensor,
    att: MultiHeadAttention,
}

impl AttentionPool2d {
    pub fn new(
        spatial_dim: usize,
        embed_dim: usize,
        n_head: usize,
        output_dim: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        info!("Load {}", vb.prefix());

        let positional_embedding =
            vb.get((spatial_dim.pow(2) + 1, embed_dim), "positional_embedding")?;
        let att = MultiHeadAttention::new(embed_dim, output_dim, n_head, vb)?;

        Ok(Self {
            positional_embedding,
            att,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.flatten_from(2)?.permute([2, 0, 1])?; // NCHW -> (HW)NC
        let mean = x.mean(0)?;
        let x = Tensor::cat(&[mean, x], 0)?; // (HW+1)NC
        let x = x.add(&self.positional_embedding.unsqueeze(1)?)?;

        let x = self.att.forward(&x.narrow(0, 0, 1)?, None)?;
        let x = x.squeeze(0)?;

        Ok(x)
    }
}

pub struct ModifiedResNet {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    conv3: Conv2d,
    bn3: BatchNorm,
    layer1: Sequence,
    layer2: Sequence,
    layer3: Sequence,
    layer4: Sequence,
    attnpool: AttentionPool2d,
}

impl ModifiedResNet {
    pub fn new(
        layers: [usize; 4],
        output_dim: usize,
        heads: usize,
        input_resolution: usize,
        width: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        info!("Load {}", vb.prefix());

        let conv1 = conv2d_no_bias(
            3,
            width / 2,
            3,
            Conv2dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let bn1 = batch_norm(width / 2, BatchNormConfig::default(), vb.pp("bn1"))?;
        let conv2 = conv2d_no_bias(
            width / 2,
            width / 2,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;
        let bn2 = batch_norm(width / 2, BatchNormConfig::default(), vb.pp("bn2"))?;
        let conv3 = conv2d_no_bias(
            width / 2,
            width,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv3"),
        )?;
        let bn3 = batch_norm(width, BatchNormConfig::default(), vb.pp("bn3"))?;

        let embed_dim = width * 32; // ResNet feature dimension
        let attnpool = AttentionPool2d::new(
            input_resolution / 32,
            embed_dim,
            heads,
            Some(output_dim),
            vb.pp("attnpool"),
        )?;

        let mut inplanes = width;

        let mut create_bottlenecks = |planes, blocks, stride| -> Result<Sequence> {
            let mut seq = Sequence::new();

            seq.add(Bottleneck::new(inplanes, planes, stride, vb.clone())?);
            inplanes = planes * EXPANSION;

            for _ in 1..blocks {
                // Lambda function for creating bottlenecks

                seq.add(Bottleneck::new(inplanes, width * 4, 1, vb.clone())?);
            }

            Ok(seq)
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            layer1: create_bottlenecks(width, layers[0], 1)?,
            layer2: create_bottlenecks(width * 2, layers[1], 2)?,
            layer3: create_bottlenecks(width * 4, layers[2], 2)?,
            layer4: create_bottlenecks(width * 8, layers[3], 2)?,
            attnpool,
        })
    }
}

impl Module for ModifiedResNet {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self
            .conv1
            .forward(input)?
            .apply_t(&self.bn1, false)?
            .relu()?;
        let x = self.conv2.forward(&x)?.apply_t(&self.bn2, false)?.relu()?;
        let x = self.conv3.forward(&x)?.apply_t(&self.bn3, false)?.relu()?;
        let x = x.avg_pool2d(2)?;

        let x = self.layer1.forward(&x)?;
        let x = self.layer2.forward(&x)?;
        let x = self.layer3.forward(&x)?;
        let x = self.layer4.forward(&x)?;

        self.attnpool.forward(&x)
    }
}

pub struct Mlp {
    c_fc: Linear,
    gelu: Activation,
    c_proj: Linear,
}

impl Mlp {
    fn new(
        input_dim: usize,
        intermediate_dim: usize,
        output_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        info!("Load {}", vb.prefix());

        let c_fc = linear(input_dim, intermediate_dim, vb.pp("c_fc"))?;
        let c_proj = linear(intermediate_dim, output_dim, vb.pp("c_proj"))?;
        Ok(Self {
            c_fc,
            gelu: Activation::QuickGelu {},
            c_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.c_fc.forward(x)?;
        let x = self.gelu.forward(&x)?;
        self.c_proj.forward(&x)
    }
}

pub struct ResidualAttentionBlock {
    attn: MultiHeadAttention,
    ln_1: LayerNorm,
    mlp: Mlp,
    ln_2: LayerNorm,
    attn_mask: Option<Tensor>,
}

impl ResidualAttentionBlock {
    fn new(
        d_model: usize,
        n_head: usize,
        attn_mask: Option<Tensor>,
        _device: &Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        info!("Load {}", vb.prefix());

        let attn = MultiHeadAttention::new(d_model, None, n_head, vb.pp("attn"))?;

        let ln_1 = LayerNorm::new(
            vb.get(&[d_model], "ln_1.weight")?,
            vb.get(&[d_model], "ln_1.bias")?,
            1e-5,
        );
        let mlp = Mlp::new(d_model, d_model * 4, d_model, vb.pp("mlp"))?;
        let ln_2 = LayerNorm::new(
            vb.get(&[d_model], "ln_2.weight")?,
            vb.get(&[d_model], "ln_2.bias")?,
            1e-5,
        );

        Ok(Self {
            attn,
            ln_1,
            mlp,
            ln_2,
            attn_mask,
        })
    }

    fn attention(&self, x: &Tensor) -> Result<Tensor> {
        let attn_mask = if let Some(ref mask) = self.attn_mask {
            Some(mask.to_dtype(x.dtype())?)
        } else {
            None
        };
        self.attn.forward(x, attn_mask.as_ref())
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.add(&self.attention(&self.ln_1.forward(x)?)?)?;
        let x = x.add(&self.mlp.forward(&self.ln_2.forward(&x)?)?)?;
        Ok(x)
    }
}

pub struct Transformer {
    width: usize,
    resblocks: Vec<ResidualAttentionBlock>,
}

impl Transformer {
    fn new(
        width: usize,
        layers: usize,
        heads: usize,
        attn_mask: Option<Tensor>,
        device: &Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        info!("Load {}", vb.prefix());

        let mut resblocks = Vec::new();
        for idx in 0..layers {
            resblocks.push(ResidualAttentionBlock::new(
                width,
                heads,
                attn_mask.clone(),
                device,
                vb.pp(format!("resblocks.{}", idx)),
            )?);
        }
        Ok(Self { width, resblocks })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.resblocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

pub struct VisionTransformer {
    input_resolution: usize,
    output_dim: usize,
    conv1: Conv2d,
    class_embedding: Tensor,
    positional_embedding: Tensor,
    ln_pre: LayerNorm,
    transformer: Transformer,
    ln_post: LayerNorm,
    proj: Tensor,
}

impl VisionTransformer {
    fn new(
        input_resolution: usize,
        patch_size: usize,
        width: usize,
        layers: usize,
        heads: usize,
        output_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        info!("Load {}", vb.prefix());

        let conv1 = conv2d_no_bias(
            3,
            width,
            patch_size,
            Conv2dConfig {
                stride: patch_size,
                padding: 0,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let scale = (width as f64).powf(-0.5);
        let class_embedding = (vb.get(&[width], "class_embedding")? * scale)?;
        let positional_embedding = (vb.get(
            &[(input_resolution / patch_size).pow(2) + 1, width],
            "positional_embedding",
        )? * scale)?;

        let weight_1 = Tensor::ones(&[width], DType::F32, &vb.device())?;
        let bias_1 = Tensor::zeros(&[width], DType::F32, &vb.device())?;
        let ln_pre = LayerNorm::new(weight_1, bias_1, 1e-5);

        let transformer = Transformer::new(
            width,
            layers,
            heads,
            None,
            &vb.device(),
            vb.pp("transformer"),
        )?;

        let weight_2 = Tensor::ones(&[width], DType::F32, &vb.device())?;
        let bias_2 = Tensor::zeros(&[width], DType::F32, &vb.device())?;
        let ln_post = LayerNorm::new(weight_2, bias_2, 1e-5);

        let proj = (vb.get(&[width, output_dim], "proj")? * scale)?;

        Ok(Self {
            input_resolution,
            output_dim,
            conv1,
            class_embedding,
            positional_embedding,
            ln_pre,
            transformer,
            ln_post,
            proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?;
        let x_shape = x.shape().dims();
        let x = x.reshape(&[x_shape[0], x_shape[1], x_shape[2] * x_shape[3]])?;
        let x = x.permute([0, 2, 1])?;

        let class_embedding = self.class_embedding.to_dtype(x.dtype())?.unsqueeze(0)?;
        let zeros = Tensor::zeros(
            &[x.shape().dims()[0], 1, x.shape().dims()[2]],
            x.dtype(),
            x.device(),
        )?;
        let class_embedding = (class_embedding.repeat(&[x.shape().dims()[0], 1, 1])? + zeros)?;
        let x = Tensor::cat(&[class_embedding, x], 1)?;

        let positional_embedding = self.positional_embedding.to_dtype(x.dtype())?;
        let x = x.broadcast_add(&positional_embedding)?;

        let x = self.ln_pre.forward(&x)?;

        let x = x.permute([1, 0, 2])?;
        let x = self.transformer.forward(&x)?;
        let x = x.permute([1, 0, 2])?;

        let x = self.ln_post.forward(&x.narrow(1, 0, 1)?.squeeze(1)?)?;

        if self.proj.shape().dims().len() == 2
            && self.proj.shape().dims()[0] == self.transformer.width
            && self.proj.shape().dims()[1] == self.output_dim
        {
            let x = x.matmul(&self.proj)?;
            Ok(x)
        } else {
            Ok(x)
        }
    }
}

///////////////////////////////////////////////////////////////////////

pub enum Visual {
    ResNet(ModifiedResNet),
    ViT(VisionTransformer),
}

impl Visual {
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        match self {
            Visual::ResNet(resnet) => resnet.forward(image),
            Visual::ViT(vit) => vit.forward(image),
        }
    }
}

pub enum VisionLayers {
    ResNet([usize; 4]),
    ViT(usize),
}

pub struct CLIP {
    context_length: usize,
    visual: Visual,
    transformer: Transformer,
    vocab_size: usize,
    token_embedding: Embedding,
    positional_embedding: Tensor,
    ln_final: LayerNorm,
    text_projection: Tensor,
    logit_scale: Tensor,
}

impl CLIP {
    fn new(
        embed_dim: usize,
        image_resolution: usize,
        vision_layers: VisionLayers,
        vision_width: usize,
        vision_patch_size: usize,
        context_length: usize,
        vocab_size: usize,
        transformer_width: usize,
        transformer_heads: usize,
        transformer_layers: usize,
        device: &Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vision_heads = vision_width * 32 / 64;

        let visual = match vision_layers {
            VisionLayers::ResNet(vision_layers) => Visual::ResNet(ModifiedResNet::new(
                vision_layers,
                embed_dim,
                vision_heads,
                image_resolution,
                vision_width,
                vb.pp("visual"),
            )?),
            VisionLayers::ViT(vision_layers) => Visual::ViT(VisionTransformer::new(
                image_resolution,
                vision_patch_size,
                vision_width,
                vision_layers,
                vision_heads,
                embed_dim,
                vb.pp("visual"),
            )?),
        };

        let transformer = Transformer::new(
            transformer_width,
            transformer_layers,
            transformer_heads,
            Some(Self::build_attention_mask(context_length, &device)?),
            device,
            vb.pp("transformer"),
        )?;

        let token_embedding =
            candle_nn::embedding(vocab_size, transformer_width, vb.pp("token_embedding"))?;

        let positional_embedding =
            vb.get(&[context_length, transformer_width], "positional_embedding")?;

        let ln_final = LayerNorm::new(
            vb.get(&[transformer_width], "ln_final.weight")?,
            vb.get(&[transformer_width], "ln_final.bias")?,
            1e-5,
        );

        let text_projection = vb.get(&[transformer_width, embed_dim], "text_projection")?;
        let logit_scale = vb.get(&[], "logit_scale")?;

        Ok(Self {
            context_length,
            visual,
            transformer,
            vocab_size,
            token_embedding,
            positional_embedding,
            ln_final,
            text_projection,
            logit_scale,
        })
    }

    pub fn build_attention_mask(size: usize, device: &Device) -> Result<Tensor> {
        //let mask = Tensor::full(f32::NEG_INFINITY, (size, size), &device)?
        //    .mul(&Tensor::triu2(size, DType::F32, &device)?)?;

        let mut mask = Vec::new();
        mask.reserve(size * size);

        for i in 0..size {
            for j in 0..size {
                if i < j {
                    mask.push(f32::NEG_INFINITY);
                } else {
                    mask.push(0.);
                }
            }
        }

        let mask = Tensor::from_vec(mask, (size, size), device)?.to_dtype(DType::F32)?;
        Ok(mask)
    }

    pub fn encode_image(&self, image: &Tensor) -> Result<Tensor> {
        self.visual.forward(image)
    }

    pub fn encode_text(&self, text: &Tensor) -> Result<Tensor> {
        let x = self.token_embedding.forward(text)?;

        let new_shape = x.shape().clone();
        let x = (x + &self.positional_embedding.expand(new_shape)?)?;

        let x = x.permute([1, 0, 2])?;
        let x = self.transformer.forward(&x)?;

        let x = x.permute([1, 0, 2])?;
        let x = self.ln_final.forward(&x)?;

        let n_batch = x.dims()[0];
        let n_dims = x.dims()[2];

        let am = text
            .argmax_keepdim(D::Minus1)?
            .unsqueeze(2)?
            .broadcast_as((n_batch, 1, n_dims))?
            .contiguous()?;
        let x = x.gather(&am, 1)?.squeeze(1)?;

        let x = x.matmul(&self.text_projection)?;

        Ok(x)
    }

    pub fn forward(&self, image: &Tensor, text: &Tensor) -> Result<(Tensor, Tensor)> {
        let image_features = self.encode_image(image)?;
        let text_features = self.encode_text(text)?;

        let image_features = norm(&image_features)?;
        let text_features = norm(&text_features)?;

        let logit_scale = self.logit_scale.exp()?;

        let logits_per_image =
            logit_scale.broadcast_mul(&image_features.matmul(&text_features.t()?)?)?;
        let logits_per_text = logits_per_image.t()?;

        Ok((logits_per_image, logits_per_text))
    }
}

pub fn build_model(device: &Device, model_path: &Path) -> Result<CLIP> {
    let map = VarMap::new();

    let embed_dim;
    let context_length;
    let vocab_size;
    let transformer_width;
    let transformer_heads;
    let transformer_layers;
    let vision_width;
    let vision_layers;
    let vision_patch_size;
    let image_resolution;

    {
        let mut vars = map.data().lock().unwrap();

        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[model_path])? };
        for (name, _) in tensors.tensors() {
            let mut tensor = tensors.load(&name, &device)?;
            //info!("Tensor {:?} {:?} {}", tensor.shape(), tensor.dtype(), name);
            if tensor.dtype() == DType::F16 {
                tensor = tensor.to_dtype(DType::F32)?;
            }
            vars.insert(name, Var::from_tensor(&tensor)?);
        }

        embed_dim = vars["text_projection"].shape().dims()[1];
        context_length = vars["positional_embedding"].shape().dims()[0];
        vocab_size = vars["token_embedding.weight"].shape().dims()[0];
        transformer_width = vars["ln_final.weight"].shape().dims()[0];
        transformer_heads = transformer_width / 64;
        transformer_layers = vars
            .keys()
            .filter(|k| k.starts_with("transformer.resblocks"))
            .map(|k| k.split('.').nth(2).unwrap())
            .collect::<std::collections::HashSet<&str>>()
            .len();

        if vars.contains_key("visual.proj") {
            vision_width = vars["visual.conv1.weight"].shape().dims()[0];
            vision_layers = VisionLayers::ViT(
                vars.keys()
                    .filter(|k| k.starts_with("visual.") && k.ends_with(".attn.in_proj_weight"))
                    .count(),
            );
            vision_patch_size = vars["visual.conv1.weight"].shape().dims()[3];
            let grid_size = ((vars["visual.positional_embedding"].shape().dims()[0] - 1) as f64)
                .sqrt() as usize;
            image_resolution = vision_patch_size * grid_size;
        } else {
            vision_width = vars["visual.layer1.0.conv1.weight"].shape().dims()[0];
            vision_layers = VisionLayers::ResNet([
                vars.keys()
                    .filter(|k| k.starts_with("visual.layer1"))
                    .count(),
                vars.keys()
                    .filter(|k| k.starts_with("visual.layer2"))
                    .count(),
                vars.keys()
                    .filter(|k| k.starts_with("visual.layer3"))
                    .count(),
                vars.keys()
                    .filter(|k| k.starts_with("visual.layer4"))
                    .count(),
            ]);
            let output_width = ((vars["visual.attnpool.positional_embedding"].shape().dims()[0] - 1)
                as f64)
                .sqrt() as usize;
            vision_patch_size = 0; // This value is not used for ResNet, so it's set to 0
            image_resolution = output_width * 32
        }
    }

    let vb = candle_nn::VarBuilder::from_varmap(&map, candle_core::DType::F32, &device);

    let model = CLIP::new(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        device,
        vb,
    )?;

    Ok(model)
}
