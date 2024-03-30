use std::path::Path;

use candle_core::{utils::cuda_is_available, DType, Device, Tensor};

use anyhow::Result;
use candle_nn::ops::softmax_last_dim;
use image::imageops::FilterType;
use tracing::warn;

mod clip;

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else {
        warn!("CUDA not available, using CPU");
        Ok(Device::Cpu)
    }
}

struct Tokenizer {
    tokenizer: tokenizers::Tokenizer,
    pad_id: u32,
}

impl Tokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Error loading tokenizer: {}", e))?;
        //let pad_id = *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap();
        let pad_id = 0;
        Ok(Self { tokenizer, pad_id })
    }

    pub fn encode(&self, prompt: &str, truncate: bool) -> Result<Tensor> {
        let mut tokens = self
            .tokenizer
            .encode(prompt, truncate)
            .map_err(|e| anyhow::anyhow!("Error encoding prompt: {}", e))?
            .get_ids()
            .to_vec();

        while tokens.len() < 77 {
            tokens.push(self.pad_id)
        }

        let tokens = Tensor::new(tokens.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
        Ok(tokens)
    }

    pub fn encode_batch(&self, prompts: &[&str], truncate: bool) -> Result<Tensor> {
        let tokens = prompts
            .iter()
            .map(|prompt| self.encode(prompt, truncate))
            .collect::<Result<Vec<_>>>()?;
        let tokens = Tensor::cat(&tokens, 0)?;
        Ok(tokens)
    }
}

fn load_image<T: AsRef<std::path::Path>>(path: T) -> Result<Tensor> {
    let img = image::io::Reader::open(path)?.decode()?;
    let img = img.resize_to_fill(224, 224, FilterType::Triangle);
    let img = img.to_rgb8().into_raw();
    let img = Tensor::from_vec(img, (224, 224, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?
        .unsqueeze(0)?;
    Ok(img)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let device = device(false)?;

    let input_text = vec!["a photo of a cat", "a photo of a dog"];
    let input_image = load_image("data/test.jpg")?;

    // Get this from https://huggingface.co/openai/clip-vit-large-patch14
    let tokenizer = Tokenizer::from_file("model/clip-vit-large-patch14/tokenizer.json")?;
    let input_text_tkn = tokenizer.encode_batch(&input_text, true)?;

    let clip = clip::build_model(
        &device,
        Path::new("model/clip-vit-large-patch14/model.safetensors"),
    )?;

    let (logits_per_image, _logits_per_text) = clip.forward(&input_image, &input_text_tkn)?;
    let softmax = softmax_last_dim(&logits_per_image)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    for (i, text) in input_text.iter().enumerate() {
        let score = softmax[i];
        println!("{}: {}", text, score);
    }

    Ok(())
}
