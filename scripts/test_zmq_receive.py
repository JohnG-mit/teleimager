#!/usr/bin/env python3
"""
Test if ZMQ data is actually being published on the server ports.
This bypasses all decoding logic to check raw data flow.
"""

import zmq
import time
import argparse
import sys

def test_port(host, port, timeout=5):
    """Test if we can receive any data from a ZMQ port."""
    print(f"\n{'='*60}")
    print(f"Testing ZMQ port {host}:{port}")
    print(f"{'='*60}")
    
    try:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)  # timeout in milliseconds
        socket.connect(f"tcp://{host}:{port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        print(f"Connected to tcp://{host}:{port}")
        print(f"Waiting for data (timeout: {timeout}s)...")
        
        start_time = time.time()
        try:
            data = socket.recv()
            elapsed = time.time() - start_time
            print(f"✓ SUCCESS! Received {len(data)} bytes in {elapsed:.2f}s")
            print(f"  First 100 bytes (hex): {data[:100].hex()}")
            return True
        except zmq.Again:
            print(f"✗ TIMEOUT! No data received after {timeout}s")
            print(f"  This means:")
            print(f"  - Server is NOT publishing data on this port")
            print(f"  - Or camera failed to initialize on server side")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False
    finally:
        try:
            socket.close()
            context.term()
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description='Test ZMQ data reception')
    parser.add_argument('--host', type=str, default='192.168.123.164', help='Server IP')
    parser.add_argument('--ports', type=int, nargs='+', default=[55555, 55558], 
                        help='Ports to test (default: 55555 55558)')
    parser.add_argument('--timeout', type=int, default=5, help='Timeout in seconds')
    args = parser.parse_args()

    print(f"ZMQ Data Reception Test")
    print(f"Server: {args.host}")
    print(f"Testing ports: {args.ports}")
    
    results = {}
    for port in args.ports:
        results[port] = test_port(args.host, port, args.timeout)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for port, success in results.items():
        status = "✓ RECEIVING DATA" if success else "✗ NO DATA"
        print(f"Port {port}: {status}")
    
    if not any(results.values()):
        print(f"\n⚠ NO DATA RECEIVED ON ANY PORT!")
        print(f"\nTroubleshooting steps:")
        print(f"1. Check server terminal output - look for:")
        print(f"   '[Image Server] head_camera is ready.'")
        print(f"   '[Image Server] Started ZMQ publisher for head_camera'")
        print(f"2. If camera is NOT ready, check:")
        print(f"   - Camera is physically connected")
        print(f"   - Camera permissions: ls -la /dev/video*")
        print(f"   - For RealSense: lsusb | grep Intel")
        print(f"3. Restart server with verbose output:")
        print(f"   python -m teleimager.image_server --rs")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
