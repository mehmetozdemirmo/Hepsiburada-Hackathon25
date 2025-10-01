import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import re
import unicodedata
from rapidfuzz import fuzz
import logging
from tqdm import tqdm
import time
import gc


class FastAddressMatcher:
    def __init__(self, n_neighbors: int = 7):
        self.setup_logging()
        self.vectorizer = None
        self.train_embeddings = None
        self.train_labels = None
        self.train_addresses = None
        self.nn_model = None
        self.n_neighbors = n_neighbors

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    # ----------------------------
    # Text preprocessing
    # ----------------------------
    def preprocess_address(self, address: str) -> str:
        """Fast CPU-based address preprocessing without GPU dependency."""
        if not address or pd.isna(address):
            return ""
        address = str(address).lower().strip()
        address = unicodedata.normalize('NFKC', address)
        address = re.sub(r'[^\w\sğüşıöçĞÜŞIÖÇ]', ' ', address)
        address = re.sub(r'\s+', ' ', address)
        # Remove duplicate words while keeping the first occurrence
        words = address.split()
        seen = set()
        cleaned_words = []
        for w in words:
            if w not in seen:
                cleaned_words.append(w)
                seen.add(w)
        return ' '.join(cleaned_words).strip()

    def extract_key_features(self, address: str) -> str:
        """Extract numeric and keyword-based features for better matching."""
        features = []
        numbers = re.findall(r'\d+', address)
        features.extend(numbers)
        keywords = [
            'mahalle', 'mahallesi', 'sokak', 'sokağı', 'cadde', 'caddesi',
            'site', 'sitesi', 'blok', 'kat', 'daire', 'numara'
        ]
        for kw in keywords:
            if kw in address:
                features.append(kw)
        # Example location-specific keywords (expand as needed)
        for loc in ['muğla', 'milas', 'aydınlıkevler', 'gümüşlük']:
            if loc in address:
                features.append(loc)
        return ' '.join(features)

    def create_enhanced_features(self, raw_address: str) -> str:
        """Generate enriched text features using n-grams and key attributes."""
        original = self.preprocess_address(raw_address)
        key_features = self.extract_key_features(original)
        words = original.split()
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)] if len(words) > 1 else []
        trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words) - 2)] if len(words) > 2 else []
        # Use only a limited number of n-grams to optimize memory and speed
        enhanced = f"{original} {key_features} {' '.join(bigrams[:3])} {' '.join(trigrams[:2])}".strip()
        return enhanced

    # ----------------------------
    # Training
    # ----------------------------
    def train_model(self, train_file: str):
        self.logger.info("Starting model training...")
        train_df = pd.read_csv(train_file)
        self.logger.info(f"Training data loaded: {len(train_df)} records")

        train_addresses = []
        train_labels = []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing training data"):
            enhanced = self.create_enhanced_features(row['address'])
            if enhanced:
                train_addresses.append(enhanced)
                train_labels.append(row['label'])
        self.train_addresses = train_addresses
        self.train_labels = np.asarray(train_labels)

        # TF-IDF vectorization using float32 for lower memory usage
        self.logger.info("Performing TF-IDF vectorization...")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            analyzer='word',
            dtype=np.float32
        )
        self.train_embeddings = self.vectorizer.fit_transform(train_addresses)

        # Nearest Neighbors with brute-force cosine similarity (CPU only)
        self.logger.info("Building Nearest Neighbors model (CPU, brute-force, cosine)...")
        self.nn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, max(1, len(self.train_labels))),
            metric='cosine',
            algorithm='brute',
            n_jobs=-1
        )
        self.nn_model.fit(self.train_embeddings)
        self.logger.info("Model training completed!")

    # ----------------------------
    # Inference helpers
    # ----------------------------
    @staticmethod
    def _majority_label(labels: np.ndarray) -> str:
        uniq, cnt = np.unique(labels, return_counts=True)
        return uniq[np.argmax(cnt)]

    def _weighted_vote(self, neighbor_labels: np.ndarray, distances: np.ndarray) -> str:
        """Weighted voting based on cosine similarity."""
        sims = 1.0 - distances
        sims = np.clip(sims, 0.0, 1.0)
        scores = {}
        for lab, w in zip(neighbor_labels, sims):
            scores[lab] = scores.get(lab, 0.0) + float(w)
        return max(scores.items(), key=lambda x: x[1])[0]

    # ----------------------------
    # Single prediction
    # ----------------------------
    def predict_single_address(self, test_address: str) -> str:
        enhanced = self.create_enhanced_features(test_address)
        if not enhanced:
            return self.train_labels[0]
        emb = self.vectorizer.transform([enhanced])
        k = min(self.n_neighbors, len(self.train_labels))
        distances, indices = self.nn_model.kneighbors(emb, n_neighbors=k)
        neighbor_labels = self.train_labels[indices[0]]
        best = self._weighted_vote(neighbor_labels, distances[0])
        # Simple confidence check
        best_train_addr = self.train_addresses[indices[0][0]]
        if fuzz.ratio(enhanced, best_train_addr) < 30:
            return self._majority_label(self.train_labels)
        return best

    # ----------------------------
    # Batch prediction
    # ----------------------------
    def predict_batch(self, test_addresses, batch_size: int = 1000):
        preds = []
        k = min(self.n_neighbors, len(self.train_labels))
        for i in tqdm(range(0, len(test_addresses), batch_size), desc="Predicting"):
            batch_raw = test_addresses[i:i + batch_size]
            enhanced_batch = [self.create_enhanced_features(a) for a in batch_raw]
            valid_idx = [j for j, a in enumerate(enhanced_batch) if a]
            if not valid_idx:
                preds.extend([self.train_labels[0]] * len(batch_raw))
                continue
            valid_texts = [enhanced_batch[j] for j in valid_idx]
            emb = self.vectorizer.transform(valid_texts)
            distances, indices = self.nn_model.kneighbors(emb, n_neighbors=k)

            batch_preds = []
            for row_d, row_i, text in zip(distances, indices, valid_texts):
                neighbor_labels = self.train_labels[row_i]
                pred = self._weighted_vote(neighbor_labels, row_d)
                # If the closest neighbor is too far, fall back to majority label
                if row_d[0] > 0.8:
                    pred = self._majority_label(self.train_labels)
                batch_preds.append(pred)

            out = []
            p = 0
            for j in range(len(batch_raw)):
                if j in valid_idx:
                    out.append(batch_preds[p])
                    p += 1
                else:
                    out.append(self.train_labels[0])
            preds.extend(out)

            if i % 5000 == 0:
                gc.collect()
        return preds

    # ----------------------------
    # File I/O pipeline
    # ----------------------------
    def predict_test_file(self, test_file: str, output_file: str):
        self.logger.info(f"Reading test file: {test_file}")
        test_df = pd.read_csv(test_file)
        self.logger.info(f"Test data loaded: {len(test_df)} records")
        start = time.time()
        predictions = self.predict_batch(test_df['address_sorted'].tolist())
        self.logger.info(f"Prediction time: {time.time() - start:.2f} seconds")
        result_df = pd.DataFrame({'id': test_df['id'], 'label': predictions})
        result_df.to_csv(output_file, index=False)
        self.logger.info(f"Results saved to: {output_file}")
        self.logger.info(f"Unique label count: {result_df['label'].nunique()}")
        return result_df


def main():
    config = {
        'train_file': 'train_benzer.csv',
        'test_file': 'test_hierarchical.csv',  # id, address_sorted
        'output_file': 'submission.csv',       # id, label
        'n_neighbors': 7
    }
    try:
        print("Fast Address Matching - CPU")
        matcher = FastAddressMatcher(n_neighbors=config['n_neighbors'])
        print("Training the model...")
        t0 = time.time()
        matcher.train_model(config['train_file'])
        t_train = time.time() - t0
        print(f"Training completed ({t_train:.1f}s)")

        print("Running predictions on test data...")
        t1 = time.time()
        result_df = matcher.predict_test_file(config['test_file'], config['output_file'])
        t_pred = time.time() - t1
        print(f"Predictions completed ({t_pred:.1f}s)")

        print("\nSUMMARY:")
        print(f"Output file: {config['output_file']}")
        print(f"Total records: {len(result_df):,}")
        print(f"Unique labels: {result_df['label'].nunique()}")
        print(f"Total time: {(t_train + t_pred)/60:.1f} minutes")
        print("\nFirst 10 predictions:")
        print(result_df.head(10).to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
