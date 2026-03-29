import struct
import sys

def read_gguf_metadata(filename):
    with open(filename, 'rb') as f:
        # Read magic
        magic = struct.unpack('<I', f.read(4))[0]
        print(f"Magic: {hex(magic)}")
        
        # Read version
        version = struct.unpack('<I', f.read(4))[0]
        print(f"Version: {version}")
        
        # Read tensor count
        tensor_count = struct.unpack('<q', f.read(8))[0]
        print(f"Tensor count: {tensor_count}")
        
        # Read metadata KV count
        kv_count = struct.unpack('<q', f.read(8))[0]
        print(f"Metadata KV count: {kv_count}")

if __name__ == "__main__":
    read_gguf_metadata(sys.argv[1])
