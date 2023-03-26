use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vfmaq_f32, vgetq_lane_f32, vld1q_f32};
use std::{collections::HashMap, fs, io::Read, os::unix::prelude::FileExt, time::Instant};

use serde::Deserialize;

static mut depth: usize = 0;

struct Timer {
    start: Instant,
    message: &'static str,
}
impl Timer {
    fn new(message: &'static str) -> Self {
        unsafe {
            println!("{}+ {}", " ".repeat(depth), message);
            depth += 1;
        }

        Self {
            start: Instant::now(),
            message,
        }
    }
}

impl Drop for Timer {
    fn drop(self: &mut Timer) {
        let now = Instant::now();
        let elapsed = now - self.start;
        unsafe {
            depth -= 1;
            println!(
                "{}- {} ({:03}ms)",
                " ".repeat(depth),
                self.message,
                elapsed.as_micros() as f32 / 1000.0
            );
        }
    }
}

fn read_u64<R: std::io::Read>(reader: &mut R) -> u64 {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf).unwrap();
    u64::from_le_bytes(buf)
}

fn read_u8<R: std::io::Read>(reader: &mut R) -> u8 {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf).unwrap();
    u8::from_le_bytes(buf)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DataType {
    Float32,
    Float64,
}

impl DataType {
    fn size(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float64 => 8,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct DenseTensor<const N: usize, T> {
    data_type: DataType,
    data: Vec<T>,
    shape: [usize; N],
}

// Debug impl
impl<const N: usize, T> std::fmt::Debug for DenseTensor<N, T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DenseTensor")
            .field("data_type", &self.data_type)
            .field("data", &self.data)
            .field("shape", &self.shape)
            .finish()
    }
}

type ParameterDescriptor = (DataType, Vec<u64>, usize);
type ParameterDescriptorMap = HashMap<String, ParameterDescriptor>;

trait FromData {
    type Output;
    fn from_data(data_type: DataType, data: &[u8]) -> Vec<Self::Output>;
}

impl<const N: usize> FromData for DenseTensor<N, f32> {
    type Output = f32;
    fn from_data(data_type: DataType, data: &[u8]) -> Vec<f32> {
        if data_type != DataType::Float32 {
            panic!("Unsupported data type");
        }
        let data = data.to_vec();
        let data = unsafe { std::mem::transmute::<Vec<u8>, Vec<f32>>(data) };
        data
    }
}

// Read from file
impl<const N: usize, T> DenseTensor<N, T>
where
    Self: FromData<Output = T>,
{
    fn from_file(
        file: &mut fs::File,
        parameter_descriptor_map: &ParameterDescriptorMap,
        name: &str,
    ) -> Self {
        let entry = parameter_descriptor_map.get(name);
        if entry.is_none() {
            panic!("Parameter not found: {}", name);
        }
        let (data_type, shape, offset) = entry.unwrap();
        let size = data_type.size() * shape.iter().product::<u64>() as usize;
        let mut data = vec![0u8; size];
        file.read_exact_at(&mut data, *offset as u64).unwrap();

        let data = Self::from_data(*data_type, &data);

        Self {
            data_type: *data_type,
            data,
            shape: shape
                .iter()
                .map(|x| *x as usize)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }
}

// Zeros like and ones like
impl<const N: usize, T> DenseTensor<N, T>
where
    T: std::default::Default + Copy,
{
    fn zeros(shape: [usize; N]) -> Self {
        Self {
            data_type: DataType::Float32,
            data: vec![T::default(); shape.iter().product::<usize>()],
            shape,
        }
    }
    fn uniform(shape: [usize; N], value: T) -> Self {
        Self {
            data_type: DataType::Float32,
            data: vec![value; shape.iter().product::<usize>()],
            shape,
        }
    }
}

// Element wise addition
impl<const N: usize, T> std::ops::Add for DenseTensor<N, T>
where
    T: std::ops::Add<Output = T> + Copy,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        assert_eq!(self.shape, rhs.shape);
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(x, y)| *x + *y)
            .collect();
        Self {
            data_type: self.data_type,
            data,
            shape: self.shape,
        }
    }
}

// Mutating element wise addition
impl<const N: usize, T> std::ops::AddAssign<&Self> for DenseTensor<N, T>
where
    T: Copy + std::ops::AddAssign,
{
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(self.shape, rhs.shape);
        for i in 0..self.data.len() {
            self.data[i] += rhs.data[i];
        }
    }
}

// Gelu
trait Gelu {
    fn gelu(self) -> Self;
}

impl Gelu for f32 {
    fn gelu(self) -> Self {
        0.5 * self
            * (1.0 + f32::tanh(f32::sqrt(2.0 / 3.1415) * (self + 0.044715 * self * self * self)))
    }
}

// Mutating gelu
impl<const N: usize> DenseTensor<N, f32> {
    fn gelu_assign(&mut self) {
        for i in 0..self.data.len() {
            self.data[i] = self.data[i].gelu();
        }
    }
}

// Row accessor
impl<T> DenseTensor<2, T> {
    fn slice_x(&self, i: usize) -> &[T] {
        let start = i * self.shape[1];
        let end = start + self.shape[1];
        &self.data[start..end]
    }
    fn slice_x_mut(&mut self, i: usize) -> &mut [T] {
        let start = i * self.shape[1];
        let end = start + self.shape[1];
        &mut self.data[start..end]
    }
}

// 3d slice accessor for accessing the third dimension
impl<T> DenseTensor<3, T> {
    fn slice_xy(&self, i: usize, j: usize) -> &[T] {
        let start = i * self.shape[1] * self.shape[2] + j * self.shape[2];
        let end = start + self.shape[2];
        &self.data[start..end]
    }
    fn slice_xy_mut(&mut self, i: usize, j: usize) -> &mut [T] {
        let start = i * self.shape[1] * self.shape[2] + j * self.shape[2];
        let end = start + self.shape[2];
        &mut self.data[start..end]
    }
}

// Index accessors
impl<T> DenseTensor<1, T> {
    fn get(&self, i: usize) -> &T {
        &self.data[i]
    }
    fn get_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}

impl<T> DenseTensor<2, T> {
    fn get(&self, i: usize, j: usize) -> &T {
        &self.data[i * self.shape[1] + j]
    }
    fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        &mut self.data[i * self.shape[1] + j]
    }
}

impl<T> DenseTensor<3, T> {
    fn get(&self, i: usize, j: usize, k: usize) -> &T {
        &self.data[i * self.shape[1] * self.shape[2] + j * self.shape[2] + k]
    }
    fn get_mut(&mut self, i: usize, j: usize, k: usize) -> &mut T {
        &mut self.data[i * self.shape[1] * self.shape[2] + j * self.shape[2] + k]
    }
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline(always)]
fn dot_f32_fast(va: &[f32], vb: &[f32]) -> f32 {
    unsafe {
        let mut accs = [vdupq_n_f32(0.0); 16];

        let mut start = 0;
        let stride = 16 * 4;
        let end = (va.len() / stride) * stride;
        while start < end {
            for i in 0..16 {
                let a = vld1q_f32(va.get_unchecked(start));
                let b = vld1q_f32(vb.get_unchecked(start));
                accs[i] = vfmaq_f32(accs[i], a, b);

                start += 4;
            }
        }

        // 1 1 1 1 1 1 1 1

        for i in 0..8 {
            accs[2 * i] = vaddq_f32(accs[2 * i], accs[2 * i + 1]);
        }

        // 1 0 1 0 1 0 1 0

        for i in 0..4 {
            accs[4 * i] = vaddq_f32(accs[4 * i], accs[4 * i + 2]);
        }

        // 1 0 0 0 1 0 0 0

        for i in 0..2 {
            accs[8 * i] = vaddq_f32(accs[8 * i], accs[8 * i + 4]);
        }

        let acc = vaddq_f32(accs[0], accs[8]);

        let mut dot = vgetq_lane_f32(acc, 0)
            + vgetq_lane_f32(acc, 1)
            + vgetq_lane_f32(acc, 2)
            + vgetq_lane_f32(acc, 3);

        for i in end..va.len() {
            dot += va[i] * vb[i];
        }

        dot
    }
} // end of eval

fn softmax_f32(a: &[f32]) -> Vec<f32> {
    let max = a.iter().fold(f32::NEG_INFINITY, |acc, x| acc.max(*x));
    let sum = a.iter().map(|x| (x - max).exp()).sum::<f32>();
    a.iter().map(|x| (x - max).exp() / sum).collect()
}

// Macro taking a prefix and a variadic number of additional nesting levels x: If prefix is empty x is returned, else prefix.x is returned, x can be arbitrarily nested
// Return type is cow
macro_rules! prefix {
    ($prefix:expr, $x:expr) => {
        if $prefix.is_empty() {
            format!("{}", $x).to_string()
        } else {
            format!("{}.{}", $prefix, $x).to_string()
        }
    };
    ($prefix:expr, $x:expr, $($xs:expr),+) => {
        &prefix!(&prefix!($prefix, $x), $($xs),+)
    };
}

type DenseTensor1F = DenseTensor<1, f32>;
type DenseTensor2F = DenseTensor<2, f32>;
type DenseTensor3F = DenseTensor<3, f32>;

type DenseTensor1U64 = DenseTensor<1, u64>;
type DenseTensor2U64 = DenseTensor<2, u64>;
type DenseTensor3U64 = DenseTensor<3, u64>;

struct LayerNorm {
    weight: DenseTensor1F,
    bias: DenseTensor1F,
}

impl LayerNorm {
    fn from_file(file: &mut std::fs::File, map: &ParameterDescriptorMap, prefix: &str) -> Self {
        Self {
            weight: DenseTensor1F::from_file(file, map, &prefix!(prefix, "weight")),
            bias: DenseTensor1F::from_file(file, map, &prefix!(prefix, "bias")),
        }
    }
    fn forward(&self, input: &DenseTensor3F) -> DenseTensor3F {
        let _timer = Timer::new("LayerNorm::forward");

        let mut output = DenseTensor3F::zeros(input.shape);
        for i in 0..input.shape[0] {
            for j in 0..input.shape[1] {
                let mean = input.slice_xy(i, j).iter().sum::<f32>() / input.shape[2] as f32;

                let mut variance = 0f32;
                for k in 0..input.shape[2] {
                    let x = input.get(i, j, k) - mean;
                    variance += x * x;
                }
                variance /= input.shape[2] as f32;
                let std = variance.sqrt();
                for k in 0..input.shape[2] {
                    let x = input.get(i, j, k);
                    let y = (x - mean) / (std + 1e-5);
                    *output.get_mut(i, j, k) = y;
                }
            }
        }
        for i in 0..input.shape[0] {
            for j in 0..input.shape[1] {
                for k in 0..input.shape[2] {
                    let x = output.get(i, j, k);
                    let y = x * self.weight.get(k) + self.bias.get(k);
                    *output.get_mut(i, j, k) = y;
                }
            }
        }

        output
    }
}

struct LinearLayer {
    weight: DenseTensor2F,
    bias: DenseTensor1F,
}

impl LinearLayer {
    fn from_file(file: &mut std::fs::File, map: &ParameterDescriptorMap, prefix: &str) -> Self {
        Self {
            weight: DenseTensor2F::from_file(file, map, &prefix!(prefix, "weight")),
            bias: DenseTensor1F::from_file(file, map, &prefix!(prefix, "bias")),
        }
    }

    #[inline(always)]
    fn forward(&self, input: &DenseTensor3F) -> DenseTensor3F {
        let _timer = Timer::new("LinearLayer::forward");

        let mut output = DenseTensor3F::zeros([input.shape[0], input.shape[1], self.bias.shape[0]]);
        let [batch_size, seq_len, _in_size] = input.shape;
        let out_size = self.bias.shape[0];

        for i in 0..batch_size {
            for k in 0..out_size {
                let weight_slice = self.weight.slice_x(k);

                for j in 0..seq_len {
                    let in_slice = input.slice_xy(i, j);

                    let sum = dot_f32_fast(&in_slice, &weight_slice);
                    *output.get_mut(i, j, k) = sum + self.bias.get(k);
                }
            }
        }
        output
    }
}

#[derive(Deserialize)]
struct BertConfig {
    attention_probs_dropout_prob: f32,
    gradient_checkpointing: bool,
    hidden_act: String,
    hidden_dropout_prob: f32,
    hidden_size: usize,
    initializer_range: f32,
    intermediate_size: usize,
    layer_norm_eps: f32,
    max_position_embeddings: usize,
    model_type: String,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    pad_token_id: usize,
    position_embedding_type: String,
    transformers_version: String,
    type_vocab_size: usize,
    use_cache: bool,
    vocab_size: usize,
}

struct BertEmbeddings {
    word_embeddings: DenseTensor2F,
    position_embeddings: DenseTensor2F,
    token_type_embeddings: DenseTensor2F,
    layer_norm: LayerNorm,
}

impl BertEmbeddings {
    fn from_file(file: &mut std::fs::File, map: &ParameterDescriptorMap, prefix: &str) -> Self {
        Self {
            word_embeddings: DenseTensor2F::from_file(
                file,
                map,
                &prefix!(prefix, "word_embeddings", "weight"),
            ),
            position_embeddings: DenseTensor2F::from_file(
                file,
                map,
                &prefix!(prefix, "position_embeddings", "weight"),
            ),
            token_type_embeddings: DenseTensor2F::from_file(
                file,
                map,
                &prefix!(prefix, "token_type_embeddings", "weight"),
            ),
            layer_norm: LayerNorm::from_file(file, map, &prefix!(prefix, "LayerNorm")),
        }
    }

    fn forward_batch(
        &self,
        input_ids: &DenseTensor2U64,
        token_type_ids: &DenseTensor2U64,
    ) -> DenseTensor3F {
        let _timer = Timer::new("BertEmbeddings::forward_batch");

        let mut output = DenseTensor3F::zeros([input_ids.shape[0], input_ids.shape[1], 768]);

        for i in 0..input_ids.shape[0] {
            for j in 0..input_ids.shape[1] {
                let word_embedding = self.word_embeddings.slice_x(*input_ids.get(i, j) as usize);
                let position_embedding = self.position_embeddings.slice_x(j);
                let token_type_embedding = self
                    .token_type_embeddings
                    .slice_x(*token_type_ids.get(i, j) as usize);

                output.slice_xy_mut(i, j).copy_from_slice(
                    &word_embedding
                        .iter()
                        .zip(position_embedding.iter())
                        .zip(token_type_embedding.iter())
                        .map(|((x, y), z)| x + y + z)
                        .collect::<Vec<_>>(),
                );
            }
        }
        self.layer_norm.forward(&output)
    }
}

struct BertSelfAttention {
    num_attention_heads: usize,
    attention_head_size: usize,

    query: LinearLayer,
    key: LinearLayer,
    value: LinearLayer,
}

impl BertSelfAttention {
    fn from_file(
        num_attention_heads: usize,
        attention_head_size: usize,
        file: &mut std::fs::File,
        prefix: &str,
        map: &ParameterDescriptorMap,
    ) -> Self {
        Self {
            num_attention_heads,
            attention_head_size,
            query: LinearLayer::from_file(file, map, &prefix!(prefix, "query")),
            key: LinearLayer::from_file(file, map, &prefix!(prefix, "key")),
            value: LinearLayer::from_file(file, map, &prefix!(prefix, "value")),
        }
    }

    fn forward_batch(
        &self,
        hidden_states: &DenseTensor3F,
        attention_mask: &DenseTensor2U64,
    ) -> DenseTensor3F {
        let _timer = Timer::new("BertSelfAttention::forward_batch");

        let mixed_query_layer = self.query.forward(&hidden_states);
        let mixed_key_layer = self.key.forward(&hidden_states);
        let mixed_value_layer = self.value.forward(&hidden_states);

        let batch_size = hidden_states.shape[0];
        let seq_length = hidden_states.shape[1];

        let normalization_factor = 1f32 / (self.attention_head_size as f32).sqrt();

        let mut new_hidden_states = DenseTensor3F::zeros([
            batch_size,
            seq_length,
            self.num_attention_heads * self.attention_head_size,
        ]);

        // batch
        for i in 0..batch_size {
            // attention head
            for j in 0..self.num_attention_heads {
                let mut attention_scores = DenseTensor2F::zeros([seq_length, seq_length]);
                let attention_head_offset = j * self.attention_head_size;

                // Do attention mask

                // query
                for k in 0..seq_length {
                    let query_vec = &mixed_query_layer.slice_xy(i, k)
                        [attention_head_offset..attention_head_offset + self.attention_head_size];

                    // key
                    for l in 0..seq_length {
                        let key_vec = &mixed_key_layer.slice_xy(i, l)[attention_head_offset
                            ..attention_head_offset + self.attention_head_size];

                        let score = dot_f32(query_vec, key_vec) * normalization_factor;
                        *attention_scores.get_mut(k, l) = score;
                    }
                }

                // softmax
                for k in 0..seq_length {
                    let thing = softmax_f32(&attention_scores.slice_x(k));
                    attention_scores.slice_x_mut(k).copy_from_slice(&thing);
                }

                // value
                for k in 0..seq_length {
                    let new_value_vec = &mut new_hidden_states.slice_xy_mut(i, k)
                        [attention_head_offset..attention_head_offset + self.attention_head_size];
                    for l in 0..seq_length {
                        let score = *attention_scores.get(k, l);
                        let value_vec = &mixed_value_layer.slice_xy(i, l)[attention_head_offset
                            ..attention_head_offset + self.attention_head_size];

                        for (x, y) in new_value_vec.iter_mut().zip(value_vec.iter()) {
                            *x += score * y;
                        }
                    }
                }
            }
        }

        new_hidden_states
    }
}

struct BertLayer {
    self_attention: BertSelfAttention,
    attention_output_dense: LinearLayer,
    attention_output_layer_norm: LayerNorm,
    intermediate_dense: LinearLayer,
    output_dense: LinearLayer,
    output_layer_norm: LayerNorm,
}

impl BertLayer {
    fn from_file(
        num_attention_heads: usize,
        attention_head_size: usize,
        file: &mut std::fs::File,
        map: &ParameterDescriptorMap,
        prefix: &str,
    ) -> Self {
        Self {
            self_attention: BertSelfAttention::from_file(
                num_attention_heads,
                attention_head_size,
                file,
                &prefix!(prefix, "attention", "self"),
                map,
            ),
            attention_output_dense: LinearLayer::from_file(
                file,
                map,
                &prefix!(prefix, "attention", "output", "dense"),
            ),
            attention_output_layer_norm: LayerNorm::from_file(
                file,
                map,
                &prefix!(prefix, "attention", "output", "LayerNorm"),
            ),
            intermediate_dense: LinearLayer::from_file(
                file,
                map,
                &prefix!(prefix, "intermediate", "dense"),
            ),
            output_dense: LinearLayer::from_file(file, map, &prefix!(prefix, "output", "dense")),
            output_layer_norm: LayerNorm::from_file(
                file,
                map,
                &prefix!(prefix, "output", "LayerNorm"),
            ),
        }
    }

    fn forward_batch(
        &self,
        hidden_states: &DenseTensor3F,
        attention_mask: &DenseTensor2U64,
    ) -> DenseTensor3F {
        let _timer = Timer::new("BertLayer::forward_batch");

        let attention_output = self
            .self_attention
            .forward_batch(hidden_states, attention_mask);

        let mut attention_output_state = self.attention_output_dense.forward(&attention_output);
        attention_output_state += hidden_states;
        let attention_output_state = self
            .attention_output_layer_norm
            .forward(&attention_output_state);

        let mut intermediate_output = self.intermediate_dense.forward(&attention_output_state);
        intermediate_output.gelu_assign();

        let mut output_hidden_states = self.output_dense.forward(&intermediate_output);
        output_hidden_states += &attention_output_state;
        output_hidden_states = self.output_layer_norm.forward(&output_hidden_states);

        output_hidden_states
    }
}

struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    fn from_file(
        num_hidden_layers: usize,
        num_attention_heads: usize,
        attention_head_size: usize,
        file: &mut std::fs::File,
        map: &ParameterDescriptorMap,
        prefix: &str,
    ) -> Self {
        Self {
            layers: (0..num_hidden_layers)
                .map(|i| {
                    BertLayer::from_file(
                        num_attention_heads,
                        attention_head_size,
                        file,
                        map,
                        &prefix!(prefix, "layer", i),
                    )
                })
                .collect(),
        }
    }

    fn forward_batch(
        &self,
        hidden_states: &DenseTensor3F,
        attention_mask: &DenseTensor2U64,
    ) -> DenseTensor3F {
        let _timer = Timer::new("BertEncoder::forward_batch");

        let mut output = hidden_states.clone();
        for layer in &self.layers {
            output = layer.forward_batch(&output, attention_mask);
        }
        output
    }
}

struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
}

impl BertModel {
    fn from_file(
        config: BertConfig,
        file: &mut std::fs::File,
        map: &ParameterDescriptorMap,
        prefix: &str,
    ) -> Self {
        let num_hidden_layers = config.num_hidden_layers;
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / num_attention_heads;
        Self {
            embeddings: BertEmbeddings::from_file(file, map, &prefix!(prefix, "embeddings")),
            encoder: BertEncoder::from_file(
                num_hidden_layers,
                num_attention_heads,
                attention_head_size,
                file,
                map,
                &prefix!(prefix, "encoder"),
            ),
        }
    }

    fn forward_batch(
        &self,
        input_ids: &DenseTensor2U64,
        token_type_ids: &DenseTensor2U64,
        attention_mask: &DenseTensor2U64,
    ) -> DenseTensor3F {
        let _timer = Timer::new("BertModel::forward_batch");

        let input_hidden_state = self.embeddings.forward_batch(input_ids, token_type_ids);
        let output_hiddent_state = self
            .encoder
            .forward_batch(&input_hidden_state, attention_mask);
        output_hiddent_state
    }
}

fn main() -> Result<(), std::io::Error> {
    // Open tinybert.bin
    let file = std::fs::File::open("tinybert.bin").unwrap();
    let mut reader = std::io::BufReader::new(file);

    // parameter count (u64)
    let param_count = read_u64(&mut reader);

    // header size (u64)
    let header_size = read_u64(&mut reader) as usize;

    let mut parameter_metadata: HashMap<String, (DataType, Vec<u64>, usize)> = HashMap::new();
    let mut offset = 16 + header_size;

    let config_len = read_u64(&mut reader);
    let mut config_buf = vec![0u8; config_len as usize];
    reader.read_exact(&mut config_buf).unwrap();
    let bert_config: BertConfig = serde_json::from_str(&String::from_utf8(config_buf).unwrap())?;

    for _ in 0..param_count {
        let name_len = read_u64(&mut reader);
        let mut name_buf = vec![0u8; name_len as usize];
        reader.read_exact(&mut name_buf).unwrap();
        let name = String::from_utf8(name_buf).unwrap();

        let data_type = match read_u8(&mut reader) {
            0 => DataType::Float32,
            1 => DataType::Float64,
            _ => panic!("Invalid data type"),
        };

        let shape_len = read_u64(&mut reader);
        let mut shape_buf = vec![0u64; shape_len as usize];
        for i in 0..shape_len as usize {
            shape_buf[i] = read_u64(&mut reader);
        }

        let size = shape_buf.iter().product::<u64>() as usize;
        parameter_metadata.insert(name, (data_type, shape_buf, offset));
        offset += data_type.size() * size;
    }

    // do pread() to read the data
    let mut file = reader.into_inner();
    let model = BertModel::from_file(bert_config, &mut file, &parameter_metadata, "");

    let mut input_ids = DenseTensor2U64::zeros([1, 9]);
    input_ids
        .slice_x_mut(0)
        .copy_from_slice(&[101, 2129, 2116, 2111, 2444, 1999, 2414, 1029, 102]);

    let token_type_ids = DenseTensor2U64::zeros([1, 9]);
    let attention_mask = DenseTensor2U64::uniform([1, 9], 1);

    let start = std::time::Instant::now();
    let output = model.forward_batch(&input_ids, &token_type_ids, &attention_mask);

    println!("{:?}", output);
    println!("Took {} ms", start.elapsed().as_millis());

    Ok(())
}
