pub mod preprocessing;
pub mod model;

#[cfg(test)]
mod tests {
    use super::*;

    use preprocessing::DataPreprocessor;
    use model::{VOCAB_SIZE, SENTENCE_SIZE, train_sentiment_model};

    #[test]
    fn it_trains_the_model() {
        let curr_dir: String = match std::env::current_dir() {
            Ok(path) => path.display().to_string(),
            Err(e) => panic!("Error: {:?}", e),
        };
        let filepath: String = format!("{}/data/chats.csv", curr_dir);
        let preprocessor: DataPreprocessor = DataPreprocessor::new(&filepath);

        let (x_train, y_train, _, _, vocab_size) 
            = preprocessor.preprocess_train_test_split_data().unwrap();

        assert_eq!(VOCAB_SIZE, vocab_size as usize);
        assert_eq!(SENTENCE_SIZE, x_train[0].len());

        train_sentiment_model(x_train, y_train);
    }
}
