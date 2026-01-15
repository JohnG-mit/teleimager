#!/usr/bin/env python3
"""
Debug RGBD data decoding from ZMQ.
"""

import zmq
import cv2
import numpy as np
import argparse

def test_decode_rgbd(host, port):
    """Test decoding RGBD data from a ZMQ port."""
    print(f"\n{'='*60}")
    print(f"Testing RGBD decode from {host}:{port}")
    print(f"{'='*60}")
    
    try:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 5000)
        socket.connect(f"tcp://{host}:{port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        print(f"Waiting for data...")
        packed_data = socket.recv()
        print(f"✓ Received {len(packed_data)} bytes")
        
        # Parse header
        if len(packed_data) < 4:
            print(f"✗ Data too short: {len(packed_data)} bytes")
            return False
        
        rgb_len = int.from_bytes(packed_data[:4], 'little')
        print(f"  Header: RGB length = {rgb_len} bytes")
        print(f"  First 20 bytes (hex): {packed_data[:20].hex()}")
        
        if len(packed_data) < 4 + rgb_len:
            print(f"✗ Invalid packet: total {len(packed_data)} < header(4) + rgb({rgb_len})")
            return False
        
        depth_len = len(packed_data) - 4 - rgb_len
        print(f"  Calculated: RGB={rgb_len}, Depth={depth_len}, Total={len(packed_data)}")
        
        # Extract data
        rgb_jpeg_bytes = packed_data[4:4 + rgb_len]
        depth_png_bytes = packed_data[4 + rgb_len:]
        
        print(f"\n  RGB JPEG data:")
        print(f"    Length: {len(rgb_jpeg_bytes)} bytes")
        print(f"    First 10 bytes: {rgb_jpeg_bytes[:10].hex()}")
        print(f"    Last 10 bytes: {rgb_jpeg_bytes[-10:].hex()}")
        
        print(f"\n  Depth PNG data:")
        print(f"    Length: {len(depth_png_bytes)} bytes")
        print(f"    First 10 bytes: {depth_png_bytes[:10].hex()}")
        print(f"    Last 10 bytes: {depth_png_bytes[-10:].hex()}")
        
        # Decode RGB
        print(f"\n  Decoding RGB JPEG...")
        try:
            rgb_np = np.frombuffer(rgb_jpeg_bytes, dtype=np.uint8)
            rgb_img = cv2.imdecode(rgb_np, cv2.IMREAD_COLOR)
            if rgb_img is not None:
                print(f"    ✓ RGB decoded: shape={rgb_img.shape}, dtype={rgb_img.dtype}")
            else:
                print(f"    ✗ RGB decode failed: cv2.imdecode returned None")
                print(f"    JPEG magic bytes: {rgb_jpeg_bytes[:4].hex()} (should be ffd8ffe0 or ffd8ffe1)")
                return False
        except Exception as e:
            print(f"    ✗ RGB decode error: {e}")
            return False
        
        # Decode Depth
        print(f"\n  Decoding Depth PNG...")
        try:
            depth_np = np.frombuffer(depth_png_bytes, dtype=np.uint8)
            depth_img = cv2.imdecode(depth_np, cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                print(f"    ✓ Depth decoded: shape={depth_img.shape}, dtype={depth_img.dtype}")
                print(f"    Depth value range: min={depth_img.min()}, max={depth_img.max()}")
            else:
                print(f"    ✗ Depth decode failed: cv2.imdecode returned None")
                print(f"    PNG magic bytes: {depth_png_bytes[:8].hex()} (should be 89504e470d0a1a0a)")
                return False
        except Exception as e:
            print(f"    ✗ Depth decode error: {e}")
            return False
        
        print(f"\n✓ SUCCESS! Both RGB and Depth decoded successfully")
        return True
        
    except zmq.Again:
        print(f"✗ Timeout waiting for data")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        socket.close()
        context.term()

def main():
    parser = argparse.ArgumentParser(description='Debug RGBD decoding')
    parser.add_argument('--host', type=str, default='192.168.123.164', help='Server IP')
    parser.add_argument('--port', type=int, default=55558, help='RGBD port')
    args = parser.parse_args()
    
    success = test_decode_rgbd(args.host, args.port)
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
