pub mod preprocessing;
pub mod model;

use preprocessing::DataPreprocessor;
use model::get_prediction;

/// Sentiment Prediction
/// Make a prediction on the sentiment of whether a sentence is a chat or a task
pub fn sentiment_prediciton(sentence: &str) -> Result<f32, String> {
    let curr_dir: String = match std::env::current_dir() {
        Ok(path) => path.display().to_string(),
        Err(e) => panic!("Error: {:?}", e),
    };
    let filepath_csv: String = format!("{}/data/chats.csv", curr_dir);
    let filepath_hashmap: String = format!("{}/data/x_hash.bin", curr_dir);

    let preprocessor: DataPreprocessor = DataPreprocessor::new(&filepath_csv, &filepath_hashmap);
    let tokens: Vec<u32> = preprocessor.post_tokenize_sentence(sentence).map_err(|e| e.to_string())?;
    let y_hat: f32 = get_prediction(tokens);
    Ok(y_hat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use model::{VOCAB_SIZE, SENTENCE_SIZE, train_sentiment_model};

    #[test]
    fn it_trains_the_model() {
        let curr_dir: String = match std::env::current_dir() {
            Ok(path) => path.display().to_string(),
            Err(e) => panic!("Error: {:?}", e),
        };
        let filepath_csv: String = format!("{}/data/chats.csv", curr_dir);
        let filepath_hashmap: String = format!("{}/data/x_hash.bin", curr_dir);
        let preprocessor: DataPreprocessor = DataPreprocessor::new(&filepath_csv, &filepath_hashmap);

        let (x_train, y_train, _, _, vocab_size) 
            = preprocessor.preprocess_train_test_split_data().unwrap();

        assert_eq!(VOCAB_SIZE, vocab_size as usize);
        assert_eq!(SENTENCE_SIZE, x_train[0].len());

        train_sentiment_model(x_train, y_train);
    }

    #[test]
    fn it_makes_a_prediction() {
        let sentence: &str = "Who did you see?";
        let y_hat: f32 = sentiment_prediciton(sentence).unwrap();
        dbg!(y_hat);
    }
}
