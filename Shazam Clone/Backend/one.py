import hashlib
import numpy as np
from typing import List, Union, Optional

class FingerprintingAlgorithm:
    def __init__(self, chunk_size: int = 1024, hash_algorithm: str = 'sha256'):
        """
        Initialize the fingerprinting algorithm
        
        Args:
            chunk_size: Size of each data chunk in bytes
            hash_algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256')
        """
        self.chunk_size = chunk_size
        self.hash_algorithm = hash_algorithm
        self.supported_hashes = ['md5', 'sha1', 'sha256', 'sha512']
        
        if hash_algorithm not in self.supported_hashes:
            raise ValueError(f"Unsupported hash algorithm. Choose from {self.supported_hashes}")

    def preprocess_data(self, data: Union[str, bytes]) -> bytes:
        """
        Preprocess input data - normalize and convert to bytes
        """
        if isinstance(data, str):
            # Normalize string: lowercase and remove extra whitespace
            data = ' '.join(data.lower().split())
            return data.encode('utf-8')
        elif isinstance(data, bytes):
            return data
        else:
            raise TypeError("Data must be string or bytes")

    def chunk_data(self, data: bytes) -> List[bytes]:
        """
        Split data into chunks of specified size
        """
        chunks = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks

    def generate_chunk_hash(self, chunk: bytes) -> str:
        """
        Generate hash for a single chunk
        """
        if self.hash_algorithm == 'md5':
            return hashlib.md5(chunk).hexdigest()
        elif self.hash_algorithm == 'sha1':
            return hashlib.sha1(chunk).hexdigest()
        elif self.hash_algorithm == 'sha256':
            return hashlib.sha256(chunk).hexdigest()
        elif self.hash_algorithm == 'sha512':
            return hashlib.sha512(chunk).hexdigest()
        else:
            raise ValueError("Unsupported hash algorithm")

    def generate_fingerprint(self, data: Union[str, bytes]) -> str:
        """
        Generate fingerprint for the input data
        """
        # Step 1: Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Step 2: Chunk data
        chunks = self.chunk_data(processed_data)
        
        # Step 3: Generate hashes for each chunk
        chunk_hashes = [self.generate_chunk_hash(chunk) for chunk in chunks]
        
        # Step 4: Combine hashes to create final fingerprint
        combined_hash_string = ''.join(chunk_hashes)
        final_fingerprint = self.generate_chunk_hash(combined_hash_string.encode('utf-8'))
        
        return final_fingerprint

    def similarity_score(self, fp1: str, fp2: str) -> float:
        """
        Calculate similarity between two fingerprints using Hamming distance
        """
        if len(fp1) != len(fp2):
            raise ValueError("Fingerprints must be of same length")
        
        # Convert hex strings to binary for better comparison
        bin1 = bin(int(fp1, 16))[2:].zfill(256)
        bin2 = bin(int(fp2, 16))[2:].zfill(256)
        
        # Calculate Hamming distance
        hamming_distance = sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
        
        # Convert to similarity score (0-1)
        similarity = 1 - (hamming_distance / len(bin1))
        
        return similarity

class AdvancedFingerprinting(FingerprintingAlgorithm):
    """
    Advanced fingerprinting with sliding window and fuzzy hashing
    """
    
    def __init__(self, window_size: int = 7, base: int = 257, mod: int = 10**9+7):
        self.window_size = window_size
        self.base = base
        self.mod = mod
        super().__init__()

    def rolling_hash(self, data: bytes) -> List[int]:
        """
        Implement rolling hash for efficient substring hashing
        """
        if len(data) < self.window_size:
            return [self.generate_simple_hash(data)]
        
        hashes = []
        n = len(data)
        
        # Precompute base powers
        base_powers = [1]
        for i in range(1, self.window_size):
            base_powers.append((base_powers[-1] * self.base) % self.mod)
        
        # Compute initial window hash
        current_hash = 0
        for i in range(self.window_size):
            current_hash = (current_hash * self.base + data[i]) % self.mod
        hashes.append(current_hash)
        
        # Slide window and update hash
        for i in range(self.window_size, n):
            # Remove leftmost character
            current_hash = (current_hash - data[i - self.window_size] * base_powers[-1]) % self.mod
            # Ensure positive
            if current_hash < 0:
                current_hash += self.mod
            # Add new character
            current_hash = (current_hash * self.base + data[i]) % self.mod
            hashes.append(current_hash)
        
        return hashes

    def generate_simple_hash(self, data: bytes) -> int:
        """Simple polynomial hash"""
        hash_val = 0
        for char in data:
            hash_val = (hash_val * self.base + char) % self.mod
        return hash_val

# Example usage and testing
def main():
    # Basic fingerprinting example
    print("=== Basic Fingerprinting ===")
    fp = FingerprintingAlgorithm(chunk_size=512, hash_algorithm='sha256')
    
    # Test with sample data
    sample_text1 = "Hello, this is a sample text for fingerprinting algorithm demonstration."
    sample_text2 = "Hello, this is a slightly modified sample text for fingerprinting."
    
    fingerprint1 = fp.generate_fingerprint(sample_text1)
    fingerprint2 = fp.generate_fingerprint(sample_text2)
    
    print(f"Text 1: {sample_text1}")
    print(f"Fingerprint 1: {fingerprint1}")
    print(f"Text 2: {sample_text2}")
    print(f"Fingerprint 2: {fingerprint2}")
    
    similarity = fp.similarity_score(fingerprint1, fingerprint2)
    print(f"Similarity score: {similarity:.4f}")
    
    # Advanced fingerprinting example
    print("\n=== Advanced Fingerprinting ===")
    advanced_fp = AdvancedFingerprinting()
    
    test_data = b"hello world"
    rolling_hashes = advanced_fp.rolling_hash(test_data)
    
    print(f"Test data: {test_data}")
    print(f"Rolling hashes (first 5): {rolling_hashes[:5]}")
    
    # File fingerprinting example
    print("\n=== File Fingerprinting ===")
    def file_fingerprint(file_path: str) -> Optional[str]:
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            return fp.generate_fingerprint(file_data)
        except FileNotFoundError:
            print(f"File {file_path} not found")
            return None
    
    # Uncomment to test with actual files
    # fingerprint = file_fingerprint("example.txt")
    # if fingerprint:
    #     print(f"File fingerprint: {fingerprint}")

if __name__ == "__main__":
    main()