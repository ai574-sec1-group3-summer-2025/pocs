from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


def encode_labels(train_df, test_df, target_col):
    le = LabelEncoder()
    le.fit(train_df[target_col])
    train_y = le.transform(train_df[target_col])
    test_y = le.transform(test_df[target_col])
    num_classes = len(le.classes_)
    return train_y, test_y, num_classes, le.classes_

def get_vectorizer(tensors, seq_len):
    raw_train_text = tf.data.Dataset.from_tensor_slices(tensors)

    vectorizer = tf.keras.layers.TextVectorization(
        standardize=None,
        split="character",
        max_tokens=None,
        output_mode="int",
        output_sequence_length=seq_len
    )

    vectorizer.adapt(raw_train_text)
    vocab_size = vectorizer.vocabulary_size()

    return vectorizer, vocab_size
