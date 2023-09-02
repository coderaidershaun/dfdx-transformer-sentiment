use csv::Reader;
use ndarray::Array2;
use ndarray_csv::Array2Reader;
use tokenizers::tokenizer::Tokenizer;

use std::error::Error;
use std::fs::File;
use std::collections::HashMap;

pub struct DataPreprocessor {
  pub filepath: String
}

impl DataPreprocessor {
  pub fn new(filepath: &str) -> Self {
    Self { filepath: filepath.to_string() }
  }

  /// Encodes labels into increments for data preprocessing
  fn encode_Labels(&self, arr: Vec<Vec<u32>>) -> Vec<u32> {
    let mut unique_values: Vec<Vec<u32>> = arr.clone();
    unique_values.sort();
    unique_values.dedup();
    let mut encoding = HashMap::new();
    for (i, value) in unique_values.iter().enumerate() { encoding.insert(value, i as u32); }
    let mut encoded_values = Vec::new();
    for value in arr { encoded_values.push(*encoding.get(&value).unwrap()); }
    encoded_values
  }

  /// Converts string sentence into token array
  fn generate_tokens_with_padding(&self, arr: &Vec<String>) -> tokenizers::tokenizer::Result<Vec<Vec<u32>>> {
    let tokenizer: Tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)?;
    let mut max_len: usize = 0;
    let mut tokens: Vec<Vec<u32>> = vec!();
    for s in arr {
      let t: tokenizers::Encoding = tokenizer.encode(s.clone(), false).unwrap();
      // let tokens: &[String] = t.get_tokens();
      let ids: &[u32] = t.get_ids();
      if ids.len() > max_len { max_len = ids.len() };
      tokens.push(ids.to_vec());
    }

    // Add padding
    tokens.iter_mut().for_each(|v| {
      let diff = max_len - v.len();
      let pads = vec![0; diff];
      v.extend(pads);
    });

    Ok(tokens)
  }

  /// Loads and encode data as x for tokens and y for label encoded
  fn load_and_encode_data(&self) -> Result<(Vec<Vec<u32>>, Vec<u32>), Box<dyn Error>> {
    let file: File = File::open(&self.filepath)?;
    let mut reader: Reader<File> = Reader::from_reader(file);
    let data: Array2<String> = reader.deserialize_array2((91, 2))?;
  
    let x_strings: Vec<String> = data.column(0).to_vec();
    let y_strings: Vec<String> = data.column(1).to_vec();

    let x: Vec<Vec<u32>> = self.generate_tokens_with_padding(&x_strings).map_err(|e| e.to_string())?;
    let y: Vec<Vec<u32>> = self.generate_tokens_with_padding(&y_strings).map_err(|e| e.to_string())?;
    let y: Vec<u32> = self.encode_Labels(y);

    Ok((x, y))
  }

  /// Convert X to unique tokens to avoid using standard and large tokens
  /// Note: You could save this for looking back up later if you wanted to
  fn x_to_unique_tokens(&self, x_arr: Vec<Vec<u32>>) -> (Vec<Vec<u32>>, u32) {
    let mut x_new: Vec<Vec<u32>> = vec!();
    let mut value = 0;

    let mut x_hash: HashMap<u32, u32> = HashMap::new();
    x_hash.insert(0, value);

    let mut vocab_size: u32 = 0;
    
    for tokens in x_arr {
      let mut ids_new: Vec<u32> = vec!();
      for k in tokens {
        let new_t = match x_hash.get(&k) {
          Some(num) => *num,
          None => {
            value += 1;
            x_hash.insert(k, value);
            if value > vocab_size { vocab_size = k; }
            value
          }
        };
        ids_new.push(new_t);
      }
      x_new.push(ids_new);
    }

    vocab_size += 1;
    (x_new, vocab_size)
  }

  // Transform y into y target probs for binary classification
  fn y_to_target_probs(&self, y: Vec<u32>) -> Vec<Vec<f32>> {
    y.iter().map(|&i| {
      if i == 0 { vec!(1.0) } else { vec!(0.0) } // Note: add a zero or 1 respectively if wanting to use more than 1 neuron on the output of the model
    }).collect()
  }

  /// Perform train, test, split on your data
  fn train_test_split<T: Clone, U: Clone>(&self, x: Vec<T>, y: Vec<U>, train_size: f32) -> (Vec<T>, Vec<U>, Vec<T>, Vec<U>) {
    assert_eq!(x.len(), y.len());
    let data_len: usize = x.len();
    let train_len: usize = (data_len as f32 * train_size).floor() as usize;

    let x_train: &[T] = &x[0..train_len];
    let y_train: &[U] = &y[0..train_len];
    let x_test: &[T] = &x[train_len..x.len()];
    let y_test: &[U] = &y[train_len..x.len()];

    (x_train.to_vec(), y_train.to_vec(), x_test.to_vec(), y_test.to_vec())
  }

  /// Entry function for data preprocessing
  pub fn preprocess_train_test_split_data(&self) -> Result<(Vec<Vec<u32>>, Vec<Vec<f32>>, Vec<Vec<u32>>, Vec<Vec<f32>>, u32), Box<dyn Error>> {
    let (x, y) = self.load_and_encode_data()?;
    let data_len: usize = x.len();
    
    let (x, vocab_size) = self.x_to_unique_tokens(x);
    
    println!("Sentence Size: \t\t{}", &x[0].len());
    println!("Vocab Size: \t\t{}", &vocab_size);

    let y: Vec<Vec<f32>> = self.y_to_target_probs(y);

    let (x_train, y_train, x_test, y_test) = self.train_test_split::<Vec<u32>, Vec<f32>>(x, y, 0.8);
    assert_eq!(x_train.len() + x_test.len(), data_len);
    assert_eq!(y_train.len() + y_test.len(), data_len);

    Ok((x_train, y_train, x_test, y_test, vocab_size))
  }
}
