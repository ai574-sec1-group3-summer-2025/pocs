import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


def check_class_distribution(df, target_col):
    print("===== Class Distribution =====")
    print(df[target_col].value_counts())
    
    sns.countplot(x=target_col, data=df)
    plt.title('Class Distribution')
    plt.show()

def preprocess_data(df, show_plots=False, target_col=None):
    # Handling Missing Values
    null_summary = df.isnull().sum()
    print(f"===== Null Summary =====\n{null_summary}")
    if null_summary.any():
        print("Dropping rows with missing values...")
        df = df.dropna()

    # Checking for Duplicate Values
    duplicate_count = df.duplicated().sum()
    print(f"===== Duplicate Summary =====\nCount: {duplicate_count}")
    if duplicate_count > 0:
        print("Dropping duplicate rows...")
        df = df.drop_duplicates()

    if show_plots:
        if target_col is None:
            raise ValueError("target_col must be specified when show_plots is True")
        check_class_distribution(df, target_col)
    return df

def make_dataset(vectorizer, texts, labels, batch_size=32, shuffle=False):
    AUTOTUNE = tf.data.AUTOTUNE

    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(texts))
    ds = ds.map(
        lambda txt, lbl: (vectorizer(txt), lbl),
        num_parallel_calls=AUTOTUNE
    )
    return ds.batch(batch_size).prefetch(AUTOTUNE)