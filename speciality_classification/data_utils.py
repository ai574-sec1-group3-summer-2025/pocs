import pandas as pd
import matplotlib.pyplot as plt
import re


class DataUtils:
    def __init__(self, default_verbose = True):
        self.default_verbose = default_verbose

    def set_default_verbose(self, verbose):
        self.default_verbose = verbose

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def handle_nulls(self, df: pd.DataFrame, verbose=None):
        null_summary = df.isnull().sum()
        self.console_log(f"===== Null Summary =====\n{null_summary}", verbose)
        if null_summary.any():
            self.console_log("Dropping rows with missing values...", verbose)
            df = df.dropna()
        return df

    def handle_duplicates(self, df: pd.DataFrame, verbose=None):
        duplicate_count = df.duplicated().sum()
        self.console_log(f"===== Duplicate Summary =====\nCount: {duplicate_count}", verbose)
        if duplicate_count > 0:
            self.console_log("Dropping duplicate rows...", verbose)
            df = df.drop_duplicates()
        return df

    def class_distribution(self, data, xlabel=None, verbose=None, show_plot=False):
        self.console_log("===== Class Distribution =====", verbose)
        counts = data.value_counts()
        self.console_log(counts, verbose)

        if show_plot:
            plt.figure(figsize=(10, 8))
            counts.plot(kind='bar', color='skyblue', edgecolor='black')

            if xlabel is not None:
                plt.title(f'{xlabel} Counts')
                plt.xlabel(xlabel)
                plt.ylabel('Count')
                plt.xticks(rotation=90)
                plt.tight_layout()

            plt.show()

        return counts

    def console_log(self, message, verbose):
        if verbose is None:
            verbose = self.default_verbose
        if verbose:
            print(message)