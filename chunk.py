"""
Local chunk module for K2-Blender addon.
Provides Chunk class for reading IFF-style binary files.
This is a minimal implementation to replace the standard library chunk module
which may not be available in all Blender Python builds.
"""

import struct


class Chunk:
    """
    Class for reading IFF chunk-style binary files.
    Mimics the standard library chunk module interface.
    """
    
    def __init__(self, file, bigendian=False, align=True):
        self.file = file
        self.bigendian = bigendian
        self.align = align
        
        # Read chunk header: 4 bytes name + 4 bytes size = 8 bytes total
        self.offset = file.tell()
        
        if bigendian:
            fmt = ">4sI"
        else:
            fmt = "<4sI"
        
        try:
            data = file.read(8)
            if len(data) < 8:
                raise EOFError
            
            name_bytes, size = struct.unpack(fmt, data)
            self.name = name_bytes
            self.chunksize = size
        except:
            raise EOFError
    
    def getname(self):
        """Return the chunk name as bytes."""
        return self.name
    
    def getsize(self):
        """Return the chunk data size."""
        return self.chunksize
    
    def read(self, size=-1):
        """
        Read data from the chunk.
        If size is -1, read up to chunksize bytes.
        """
        if size == -1:
            size = self.chunksize
        
        # Don't read past chunk end
        remaining = self.chunksize - (self.file.tell() - self.offset - 8)
        if size > remaining:
            size = remaining
        
        if size <= 0:
            return b''
        
        return self.file.read(size)
    
    def skip(self):
        """
        Skip to the end of the chunk, handling alignment if needed.
        """
        # Calculate bytes remaining in chunk
        pos = self.file.tell()
        chunk_end = self.offset + 8 + self.chunksize
        
        if pos < chunk_end:
            skip_bytes = chunk_end - pos
            # IFF chunks are aligned to 2 bytes by default
            if self.align:
                if skip_bytes % 2 != 0:
                    skip_bytes += 1
            self.file.read(skip_bytes)
    
    def tell(self):
        """Return current position within chunk data."""
        return self.file.tell() - self.offset - 8
    
    def seek(self, pos, whence=0):
        """
        Seek within the chunk data.
        whence: 0=from start, 1=from current, 2=from end
        """
        if whence == 0:
            # From start of chunk data
            new_pos = self.offset + 8 + pos
        elif whence == 1:
            # From current position
            new_pos = self.file.tell() + pos
        elif whence == 2:
            # From end of chunk
            new_pos = self.offset + 8 + self.chunksize + pos
        else:
            raise ValueError("Invalid whence value")
        
        self.file.seek(new_pos)
        return self.tell()
    
    def close(self):
        """Close the chunk (no-op for our implementation)."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
