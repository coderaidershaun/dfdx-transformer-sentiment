use dfdx::nn::DeviceBuildExt;
use dfdx::losses::binary_cross_entropy_with_logits_loss;
use dfdx::shapes::Const;
use dfdx::tensor::{AutoDevice, Tensor, Trace};
use dfdx::tensor::TensorFrom;
use dfdx::tensor_ops::MeanTo;
use dfdx::nn::ZeroGrads;
use dfdx::nn::ModuleMut;
use dfdx::tensor::AsArray;
use dfdx::tensor_ops::Backward;
use dfdx::optim::Adam;
use dfdx::optim::Optimizer;

use dfdx::prelude::{SaveToSafetensors, LoadFromNpz, Module};
use dfdx::nn::SaveToNpz;

pub const EPOCHS: usize = 50;
pub const SENTENCE_SIZE: usize = 19;
pub const VOCAB_SIZE: usize = 1328; // len(tokenizer.word_index) + 1 (or rather number of unique words in data + 1)
pub const EMBEDDING_DIM: usize = 32; // Size of context for each word (Attention is All You Need paper uses 512)
pub const NUM_HEADS: usize = 4; // Num of self-attention heads to find meaning
pub const NUM_CLASSES: usize = 1; // Num of potential outcomes (Chat and Task - using 1 for binary)

pub const FF_DIM: usize = 64; // Typically larger than the Embedding Dims
pub const NUM_LAYERS: usize = 2; // How many encoders we want to stack

/// Train Model
/// Trains sentiment prediction model
pub fn train_sentiment_model(
  x_train: Vec<Vec<u32>>, 
  y_train: Vec<Vec<f32>>
) {

  let dev: dfdx::tensor::Cpu = AutoDevice::default();
  type Device = dfdx::tensor::Cpu;
  type DType = f32;

  type SentimentModel = (
    dfdx::nn::builders::Embedding<VOCAB_SIZE, EMBEDDING_DIM>,
    dfdx::nn::builders::TransformerEncoder<EMBEDDING_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS>,
    dfdx::nn::builders::Linear<SENTENCE_SIZE, NUM_CLASSES>
  );

  type EmbeddingStructure = dfdx::prelude::modules::Embedding<VOCAB_SIZE, EMBEDDING_DIM, DType, Device>;
  type EncoderStructure = dfdx::prelude::modules::TransformerEncoder<EMBEDDING_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS, DType, Device>;
  type ClassifierStructure = dfdx::prelude::modules::Linear<SENTENCE_SIZE, NUM_CLASSES, DType, Device>;

  let mut model = dev.build_module::<SentimentModel, DType>();
  let mut grads = model.alloc_grads();
  let mut opt: Adam<(EmbeddingStructure, EncoderStructure, ClassifierStructure), DType, Device> = Adam::new(&model, Default::default());

  let start: std::time::Instant = std::time::Instant::now();
  let mut total_epoch_loss: f32 = 0.0;
  for epoch in 0..EPOCHS {
    for (i, x_input) in x_train.clone().iter().enumerate() {

      let x_d: Vec<usize> = x_input.iter().map(|&d| d as usize).collect();
      let x_d: &[usize] = x_d.as_slice();
      let x_d_array: &[usize; SENTENCE_SIZE] = x_d.try_into().expect("wrong length");
      let x: Tensor<(Const<SENTENCE_SIZE>,), usize, Device> = dev.tensor(x_d_array);

      let y_d: &[f32] = y_train[i].as_slice();
      let y_d_array: &[f32; NUM_CLASSES] = y_d.try_into().expect("wrong length");
      let target_probs: Tensor<(Const<NUM_CLASSES>,), f32, Device> = dev.tensor(y_d_array);
      
      // // NOTE from Repo on ways to train:
      // // and of course forward passes the input through each module sequentially:
      // // https://github.com/coreylowman/dfdx/blob/main/examples/03-nn.rs
      // let x: Tensor<Rank1<4>, f32, _> = dev.sample_normal();
      // let a = mlp.forward(x.clone());
      // let b = mlp.2.forward(mlp.1.forward(mlp.0.forward(x)));
      // assert_eq!(a.array(), b.array());
      let x: Tensor<(Const<SENTENCE_SIZE>, Const<EMBEDDING_DIM>), DType, Device, dfdx::tensor::OwnedTape<DType, Device>> = model.0.forward_mut(x.leaky_trace());
      let x: Tensor<(Const<SENTENCE_SIZE>, Const<EMBEDDING_DIM>), DType, Device, dfdx::tensor::OwnedTape<DType, Device>> = model.1.forward_mut(x);
      let x: Tensor<(Const<SENTENCE_SIZE>,), DType, Device, dfdx::tensor::OwnedTape<DType, Device>> = x.mean(); // GlobalPoolLayer as library does not have one
      let logits: Tensor<(Const<NUM_CLASSES>,), DType, Device, dfdx::tensor::OwnedTape<DType, Device>> = model.2.forward_mut(x);

      let loss = binary_cross_entropy_with_logits_loss(logits, target_probs);
      total_epoch_loss += loss.array();
      
      grads = loss.backward();
      opt.update(&mut model, &grads).unwrap();
      model.zero_grads(&mut grads);

    }
    
    // Show loss
    let msg: String = format!("epoch: {}, loss: ,{},", epoch, total_epoch_loss);
    dbg!(msg);

    // Save model
    // https://github.com/coreylowman/dfdx/blob/main/examples/safetensors-save-load.rs
    model.save_safetensors("model.safetensors").expect("Failed to save model");
    let _ = model.save("mymodel.npz");
  }

  let end: std::time::Instant = std::time::Instant::now();
  dbg!(&start, &end);
}

/// Get Prediction
/// Makes a prediction based upon text input
pub fn get_prediction(x_sentence: Vec<u32>) -> f32 {
  let dev: dfdx::tensor::Cpu = AutoDevice::default();
  type Device = dfdx::tensor::Cpu;
  type DType = f32;

  type SentimentModel = (
    dfdx::nn::builders::Embedding<VOCAB_SIZE, EMBEDDING_DIM>,
    dfdx::nn::builders::TransformerEncoder<EMBEDDING_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS>,
    dfdx::nn::builders::Linear<SENTENCE_SIZE, NUM_CLASSES>
  );

  let mut model = dev.build_module::<SentimentModel, DType>();
  model.load("mymodel.npz").unwrap();

  let x_d: Vec<usize> = x_sentence.iter().map(|&d| d as usize).collect();
  let x_d: &[usize] = x_d.as_slice();
  let x_d_array: &[usize; SENTENCE_SIZE] = x_d.try_into().expect("wrong length");
  let x: Tensor<(Const<SENTENCE_SIZE>,), usize, Device> = dev.tensor(x_d_array);

  let x: Tensor<(Const<SENTENCE_SIZE>, Const<EMBEDDING_DIM>), DType, Device, dfdx::tensor::OwnedTape<DType, Device>> = model.0.forward(x.leaky_trace());
  let x: Tensor<(Const<SENTENCE_SIZE>, Const<EMBEDDING_DIM>), DType, Device, dfdx::tensor::OwnedTape<DType, Device>> = model.1.forward(x);
  let x: Tensor<(Const<SENTENCE_SIZE>,), DType, Device, dfdx::tensor::OwnedTape<DType, Device>> = x.mean(); // GlobalPoolLayer as library does not have one
  let logits: Tensor<(Const<NUM_CLASSES>,), DType, Device, dfdx::tensor::OwnedTape<DType, Device>> = model.2.forward(x);

  let y_hat: Tensor<(Const<NUM_CLASSES>,), DType, Device, dfdx::tensor::OwnedTape<DType, Device>> = logits.sigmoid();
  let y_hat: Vec<f32> = y_hat.as_vec();

  y_hat[0]
}
