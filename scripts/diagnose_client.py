#!/usr/bin/env python3
"""
Diagnostic script to troubleshoot image_client connection issues.
"""

import sys
import time
import socket
import argparse
import logging_mp

logger_mp = logging_mp.get_logger(__name__, level=logging_mp.INFO)

def check_server_connectivity(host, port, timeout=2):
    """Check if server port is reachable."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        logger_mp.error(f"Error checking connectivity to {host}:{port}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Diagnose image client connection issues')
    parser.add_argument('--host', type=str, default='192.168.123.164', help='Server IP address')
    parser.add_argument('--req-port', type=int, default=60000, help='Server request port')
    parser.add_argument('--zmq-ports', type=int, nargs='+', default=[55555, 55556, 55557, 55558, 55559, 55560],
                        help='ZMQ ports to check')
    args = parser.parse_args()

    logger_mp.info("=" * 70)
    logger_mp.info("Image Client Diagnostic Tool")
    logger_mp.info("=" * 70)

    # Check if host is reachable
    logger_mp.info(f"\n1. Checking basic connectivity to {args.host}...")
    if check_server_connectivity(args.host, args.req_port):
        logger_mp.info(f"   ✓ Server appears to be reachable on port {args.req_port}")
    else:
        logger_mp.error(f"   ✗ Cannot reach server on port {args.req_port}")
        logger_mp.error(f"   Please check:")
        logger_mp.error(f"   - Server IP address is correct: {args.host}")
        logger_mp.error(f"   - Server process is running: python -m teleimager.image_server")
        logger_mp.error(f"   - Network connectivity between client and server")
        return 1

    # Check if we can get the configuration
    logger_mp.info(f"\n2. Attempting to get camera configuration from server...")
    try:
        from teleimager.image_client import ZMQ_Requester
        requester = ZMQ_Requester(args.host, args.req_port)
        cam_config = requester.request()
        requester.close()
        
        if cam_config:
            logger_mp.info(f"   ✓ Successfully retrieved camera configuration")
            logger_mp.info(f"   Cameras configured:")
            for cam_name in ['head_camera', 'left_wrist_camera', 'right_wrist_camera']:
                if cam_name in cam_config:
                    cam = cam_config[cam_name]
                    enable_zmq = cam.get('enable_zmq', False)
                    zmq_port = cam.get('zmq_port', 'N/A')
                    enable_depth = cam.get('enable_depth', False)
                    depth_port = cam.get('depth_zmq_port', 'N/A')
                    logger_mp.info(f"   - {cam_name}:")
                    logger_mp.info(f"       ZMQ: {enable_zmq} (port {zmq_port})")
                    if enable_depth:
                        logger_mp.info(f"       Depth: enabled (port {depth_port})")
        else:
            logger_mp.error(f"   ✗ Failed to retrieve camera configuration")
            return 1
    except Exception as e:
        logger_mp.error(f"   ✗ Error contacting server: {e}")
        return 1

    # Check if ZMQ ports are accessible
    logger_mp.info(f"\n3. Checking ZMQ publish ports...")
    accessible_ports = []
    for port in set(args.zmq_ports):
        if check_server_connectivity(args.host, port, timeout=1):
            logger_mp.info(f"   ✓ Port {port} is accessible")
            accessible_ports.append(port)
        else:
            logger_mp.info(f"   - Port {port} is not responding yet (server may be initializing cameras)")

    if accessible_ports:
        logger_mp.info(f"   {len(accessible_ports)} ports are accessible")
    else:
        logger_mp.warning(f"   No ZMQ ports are responding.")
        logger_mp.warning(f"   This usually means:")
        logger_mp.warning(f"   - Server is still starting up")
        logger_mp.warning(f"   - Cameras are not initialized yet")
        logger_mp.warning(f"   - Server encountered an error")

    logger_mp.info(f"\n4. Recommendations:")
    logger_mp.info(f"   - If server is running, try running client again in a few moments")
    logger_mp.info(f"   - Check server logs for any error messages")
    logger_mp.info(f"   - For RealSense cameras, ensure they are properly connected")
    logger_mp.info(f"   - Run: python -m teleimager.image_server --cf")
    logger_mp.info(f"     to verify camera detection")

    logger_mp.info("\n" + "=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
