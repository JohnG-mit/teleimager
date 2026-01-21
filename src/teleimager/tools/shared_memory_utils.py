# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
A simplified multi-image shared memory tool module
When writing, concatenate three images (head, left, right) horizontally and write them
When reading, split the concatenated image into three independent images
"""

import ctypes
import time
import numpy as np
import cv2
from multiprocessing import shared_memory
from typing import Optional, Dict, List
import struct
import os

# shared memory configuration
# Use separate shared memory for each image/depth stream
def get_shm_name(image_name: str, stream: str = "image") -> str:
    """Get shared memory name for a specific image stream.

    Args:
        image_name: image key name (e.g., 'head', 'left', 'right')
        stream: 'image' for RGB, 'depth' for depth
    """
    if stream == "image":
        return f"isaac_{image_name}_image_shm"
    return f"isaac_{image_name}_{stream}_shm"

SHM_SIZE_PER_IMAGE = 640 * 480 * 3 + 128  # ~1MB per image + header + buffer
SHM_SIZE_PER_DEPTH = 640 * 480 * 2 + 128  # ~0.6MB per depth + header + buffer

# Backward compatibility
SHM_NAME = "isaac_multi_image_shm"  # Kept for backward compatibility
SHM_SIZE = SHM_SIZE_PER_IMAGE * 3   # Kept for backward compatibility

# define the simplified header structure
class SimpleImageHeader(ctypes.LittleEndianStructure):  # Use little-endian for cross-platform compatibility
    """Simplified image header structure for individual images"""
    _fields_ = [
        ('timestamp', ctypes.c_uint64),    # timestamp
        ('height', ctypes.c_uint32),       # image height
        ('width', ctypes.c_uint32),        # image width
        ('channels', ctypes.c_uint32),     # number of channels
        ('image_name', ctypes.c_char * 16), # image name (e.g., 'head', 'left', 'right')
        ('data_size', ctypes.c_uint32),    # data size
        ('encoding', ctypes.c_uint32),     # 0=raw BGR, 1=JPEG
        ('quality', ctypes.c_uint32),      # JPEG quality (valid if encoding=1)
    ]


class MultiImageWriter:
    """A simplified multi-image shared memory writer using separate SHM for each image"""

    def __init__(
            self, 
            enable_jpeg: bool = False, 
            jpeg_quality: int = 85, 
            skip_cvtcolor: bool = False
        ):
        """Initialize the multi-image shared memory writer

        Args:
            enable_jpeg: whether to enable JPEG compression
            jpeg_quality: JPEG quality (0-100)
            skip_cvtcolor: whether to skip color conversion
        """
        # 50 FPS 限速（避免高频阻塞主循环）
        self._min_interval_sec = 1.0 / 50.0
        self._last_write_ts_ms = 0

        # 压缩与颜色空间配置（由主进程注入）
        self._enable_jpeg = bool(enable_jpeg)
        self._jpeg_quality = int(jpeg_quality)
        self._skip_cvtcolor = bool(skip_cvtcolor)

        # 为每个图像维护独立的共享内存
        self.shms = {}  # image_name -> SharedMemory
        print(f"[MultiImageWriter] Initialized with separate SHM per image")

    def set_options(self, *, enable_jpeg: Optional[bool] = None, jpeg_quality: Optional[int] = None, skip_cvtcolor: Optional[bool] = None):
        if enable_jpeg is not None:
            self._enable_jpeg = bool(enable_jpeg)
        if jpeg_quality is not None:
            self._jpeg_quality = int(jpeg_quality)
        if skip_cvtcolor is not None:
            self._skip_cvtcolor = bool(skip_cvtcolor)

    def _get_or_create_shm(self, image_name: str, *, stream: str, shm_size: int) -> shared_memory.SharedMemory:
        shm_name = get_shm_name(image_name, stream=stream)
        if shm_name not in self.shms:
            try:
                self.shms[shm_name] = shared_memory.SharedMemory(name=shm_name)
            except FileNotFoundError:
                self.shms[shm_name] = shared_memory.SharedMemory(create=True, size=shm_size, name=shm_name)
        return self.shms[shm_name]

    def _write_buffer(self, *, shm: shared_memory.SharedMemory, header: "SimpleImageHeader", data_bytes: bytes) -> bool:
        header.data_size = len(data_bytes)
        header_size = ctypes.sizeof(SimpleImageHeader)
        total_size = header_size + header.data_size
        if total_size > shm.size:
            print(f"[MultiImageWriter] Not enough space for {header.image_name.decode('utf-8').rstrip(chr(0))}: need {total_size}, available {shm.size}")
            return False

        header_bytes = ctypes.string_at(ctypes.byref(header), header_size)
        shm.buf[0:header_size] = header_bytes
        data_start = header_size
        data_end = data_start + header.data_size
        shm.buf[data_start:data_end] = data_bytes
        return True

    def write_images(self, images: Dict[str, np.ndarray]) -> bool:
        """Write multiple images to separate shared memories

        Args:
            images: the image dictionary, the key is the image name ('head', 'left', 'right'), the value is the image array

        Returns:
            bool: whether the writing is successful
        """
        if not images:
            return False

        # 轻量限速：最多 50 FPS，直接跳过多余写入，避免阻塞主循环
        now_ms = int(time.time() * 1000)
        if self._last_write_ts_ms and (now_ms - self._last_write_ts_ms) < int(self._min_interval_sec * 1000):
            return True

        success_count = 0

        for image_name, image in images.items():
            try:
                shm = self._get_or_create_shm(image_name, stream="image", shm_size=SHM_SIZE_PER_IMAGE)

                # 确保连续内存布局，尽量减少拷贝
                if not image.flags['C_CONTIGUOUS']:
                    image = np.ascontiguousarray(image)
                # OpenCV 期望 BGR 格式；可通过配置跳过转换
                if image.ndim == 3 and image.shape[2] == 3:
                    if not self._skip_cvtcolor:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # get the image information
                height, width, channels = image.shape

                # 准备头部
                header = SimpleImageHeader()
                header.timestamp = now_ms  # millisecond timestamp
                header.height = height
                header.width = width
                header.channels = channels
                header.image_name = image_name.encode('utf-8')[:15].ljust(16, b'\x00')  # truncate and pad to 16 bytes

                # 计算数据
                if self._enable_jpeg:
                    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self._jpeg_quality)]
                    ok, buffer = cv2.imencode('.jpg', image, encode_params)
                    if not ok:
                        print(f"[MultiImageWriter] Failed to encode {image_name} as JPEG")
                        continue
                    data_bytes = buffer.tobytes()
                    header.encoding = 1
                    header.quality = int(self._jpeg_quality)
                else:
                    data_bytes = image.tobytes()
                    header.encoding = 0
                    header.quality = 0

                if self._write_buffer(shm=shm, header=header, data_bytes=data_bytes):
                    success_count += 1

            except Exception as e:
                print(f"[MultiImageWriter] Error writing {image_name}: {e}")
                continue

        self._last_write_ts_ms = now_ms
        return success_count > 0

    def write_rgbd_pair(
        self,
        image_name: str,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        *,
        depth_encoding: str = "png"
    ) -> bool:
        """Write an RGB + depth pair into separate shared memories with the same timestamp.

        Args:
            image_name: image key (e.g., 'head')
            rgb_image: BGR/RGB image
            depth_image: depth image (uint16)
            depth_encoding: 'png' or 'raw16'
        """
        if rgb_image is None or depth_image is None:
            return False

        now_ms = int(time.time() * 1000)

        rgb_ok = False
        depth_ok = False

        try:
            # RGB write
            rgb_shm = self._get_or_create_shm(image_name, stream="image", shm_size=SHM_SIZE_PER_IMAGE)

            if not rgb_image.flags['C_CONTIGUOUS']:
                rgb_image = np.ascontiguousarray(rgb_image)
            if rgb_image.ndim == 3 and rgb_image.shape[2] == 3:
                if not self._skip_cvtcolor:
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            height, width, channels = rgb_image.shape
            rgb_header = SimpleImageHeader()
            rgb_header.timestamp = now_ms
            rgb_header.height = height
            rgb_header.width = width
            rgb_header.channels = channels
            rgb_header.image_name = image_name.encode('utf-8')[:15].ljust(16, b'\x00')

            if self._enable_jpeg:
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self._jpeg_quality)]
                ok, buffer = cv2.imencode('.jpg', rgb_image, encode_params)
                if ok:
                    rgb_data = buffer.tobytes()
                    rgb_header.encoding = 1
                    rgb_header.quality = int(self._jpeg_quality)
                else:
                    rgb_data = rgb_image.tobytes()
                    rgb_header.encoding = 0
                    rgb_header.quality = 0
            else:
                rgb_data = rgb_image.tobytes()
                rgb_header.encoding = 0
                rgb_header.quality = 0

            rgb_ok = self._write_buffer(shm=rgb_shm, header=rgb_header, data_bytes=rgb_data)
        except Exception as e:
            print(f"[MultiImageWriter] Error writing RGB {image_name}: {e}")

        try:
            # Depth write
            depth_shm = self._get_or_create_shm(image_name, stream="depth", shm_size=SHM_SIZE_PER_DEPTH)

            if not depth_image.flags['C_CONTIGUOUS']:
                depth_image = np.ascontiguousarray(depth_image)

            depth_header = SimpleImageHeader()
            depth_header.timestamp = now_ms
            depth_header.height = depth_image.shape[0]
            depth_header.width = depth_image.shape[1]
            depth_header.channels = 1 if depth_image.ndim == 2 else depth_image.shape[2]
            depth_header.image_name = image_name.encode('utf-8')[:15].ljust(16, b'\x00')

            if depth_encoding == "png":
                ok, buffer = cv2.imencode('.png', depth_image)
                if not ok:
                    raise RuntimeError("depth png encode failed")
                depth_data = buffer.tobytes()
                depth_header.encoding = 2  # PNG depth
                depth_header.quality = 0
            elif depth_encoding == "raw16":
                depth_data = depth_image.tobytes()
                depth_header.encoding = 3  # RAW16 depth
                depth_header.quality = 0
            else:
                raise ValueError(f"Unsupported depth_encoding: {depth_encoding}")

            depth_ok = self._write_buffer(shm=depth_shm, header=depth_header, data_bytes=depth_data)
        except Exception as e:
            print(f"[MultiImageWriter] Error writing depth {image_name}: {e}")

        return rgb_ok and depth_ok

    def close(self):
        """Close all shared memories"""
        for shm_name, shm in self.shms.items():
            try:
                shm.close()
                print(f"[MultiImageWriter] Shared memory closed: {shm_name}")
            except Exception as e:
                print(f"[MultiImageWriter] Error closing {shm_name}: {e}")
        self.shms.clear()


class MultiImageReader:
    """A simplified multi-image shared memory reader using separate SHM per image"""

    def __init__(self):
        """Initialize the multi-image shared memory reader"""
        self.last_timestamps = {}  # image_name -> last_timestamp
        self.last_rgbd_timestamps = {}  # image_name -> last_rgbd_timestamp
        self.buffer = {}  # image_name -> cached_image
        self.depth_buffer = {}  # image_name -> cached_depth
        self.shms = {}  # image_name -> SharedMemory

    def read_images(self) -> Optional[Dict[str, np.ndarray]]:
        """Read images from all available separate shared memories

        Returns:
            Dict[str, np.ndarray]: the image dictionary, the key is the image name, the value is the image array
        """
        images = {}
        image_names = ['head', 'left', 'right']  # Standard image names

        for image_name in image_names:
            try:
                shm_name = get_shm_name(image_name, stream="image")

                # Open shared memory if not already open
                if shm_name not in self.shms:
                    try:
                        self.shms[shm_name] = shared_memory.SharedMemory(name=shm_name)
                    except FileNotFoundError:
                        continue  # Skip if shared memory doesn't exist

                shm = self.shms[shm_name]
                header_size = ctypes.sizeof(SimpleImageHeader)

                # Read header
                header_data = bytes(shm.buf[:header_size])
                header = SimpleImageHeader.from_buffer_copy(header_data)

                # Check timestamp
                last_ts = self.last_timestamps.get(image_name, 0)
                if header.timestamp <= last_ts:
                    # Return cached image if available
                    if image_name in self.buffer:
                        images[image_name] = self.buffer[image_name]
                    continue

                # Read payload
                data_start = header_size
                data_end = data_start + header.data_size
                payload = bytes(shm.buf[data_start:data_end])

                # Decode image
                if header.encoding == 1:  # JPEG
                    encoded = np.frombuffer(payload, dtype=np.uint8)
                    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                else:  # RAW
                    image = np.frombuffer(payload, dtype=np.uint8)
                    expected_size = header.height * header.width * header.channels
                    if image.size != expected_size:
                        print(f"[MultiImageReader] Data size mismatch for {image_name}: expected {expected_size}, got {image.size}")
                        continue
                    image = image.reshape(header.height, header.width, header.channels)

                # Cache and return
                self.buffer[image_name] = image
                self.last_timestamps[image_name] = header.timestamp
                images[image_name] = image

            except Exception as e:
                print(f"[MultiImageReader] Error reading {image_name}: {e}")
                continue

        return images if images else None

    def read_concatenated_image(self) -> Optional[np.ndarray]:
        """Read all images and concatenate them horizontally (for backward compatibility)

        Returns:
            np.ndarray: the concatenated image array; if the reading fails, return None
        """
        images = self.read_images()
        if images is None or not images:
            return None

        try:
            # Concatenate images in order: head, left, right
            image_order = ['head', 'left', 'right']
            frames_to_concat = []

            for image_name in image_order:
                if image_name in images:
                    frames_to_concat.append(images[image_name])

            if not frames_to_concat:
                return None

            if len(frames_to_concat) > 1:
                concatenated_image = cv2.hconcat(frames_to_concat)
            else:
                concatenated_image = frames_to_concat[0]

            return concatenated_image

        except Exception as e:
            print(f"[MultiImageReader] Error concatenating images: {e}")
            return None

    def read_single_image(self, image_name: str) -> Optional[np.ndarray]:
        """Read a single specific image from its dedicated shared memory.

        Args:
            image_name: Name of the image to read ("head", "left", or "right")

        Returns:
            np.ndarray: The requested image array, or None if not found or error
        """
        try:
            shm_name = get_shm_name(image_name)

            # Open shared memory if not already open
            if shm_name not in self.shms:
                try:
                    self.shms[shm_name] = shared_memory.SharedMemory(name=shm_name)
                except FileNotFoundError:
                    print(f"[MultiImageReader] Shared memory {shm_name} not found")
                    return None

            shm = self.shms[shm_name]
            header_size = ctypes.sizeof(SimpleImageHeader)

            # Read header
            header_data = bytes(shm.buf[:header_size])
            header = SimpleImageHeader.from_buffer_copy(header_data)

            # Check if there is new data
            last_ts = self.last_timestamps.get(image_name, 0)
            if header.timestamp <= last_ts:
                # Return cached image if available
                return self.buffer.get(image_name)

            # Read payload
            data_start = header_size
            data_end = data_start + header.data_size
            payload = bytes(shm.buf[data_start:data_end])

            # Decode image
            if header.encoding == 1:  # JPEG
                encoded = np.frombuffer(payload, dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if image is None:
                    return None
            else:  # RAW
                image = np.frombuffer(payload, dtype=np.uint8)
                expected_size = header.height * header.width * header.channels
                if image.size != expected_size:
                    print(f"[MultiImageReader] Data size mismatch for {image_name}: expected {expected_size}, got {image.size}")
                    return None
                image = image.reshape(header.height, header.width, header.channels)

            # Update buffer and timestamp
            self.buffer[image_name] = image
            self.last_timestamps[image_name] = header.timestamp
            return image

        except Exception as e:
            print(f"[MultiImageReader] Error reading single image {image_name}: {e}")
            return None

    def read_encoded_frame(self, image_name: str = "head") -> Optional[bytes]:
        """Read encoded payload for a specific image if available (e.g., JPEG). Returns bytes or None."""
        shm_name = get_shm_name(image_name, stream="image")
        if shm_name not in self.shms:
            try:
                self.shms[shm_name] = shared_memory.SharedMemory(name=shm_name)
            except FileNotFoundError:
                return None

        shm = self.shms[shm_name]

        try:
            header_size = ctypes.sizeof(SimpleImageHeader)
            header_data = bytes(shm.buf[:header_size])
            header = SimpleImageHeader.from_buffer_copy(header_data)

            if header.encoding != 1:
                return None

            last_ts = self.last_timestamps.get(image_name, 0)
            if header.timestamp <= last_ts:
                return None

            data_start = header_size
            data_end = data_start + header.data_size
            payload = bytes(shm.buf[data_start:data_end])

            self.last_timestamps[image_name] = header.timestamp
            return payload

        except Exception as e:
            print(f"[MultiImageReader] Error reading encoded frame for {image_name}: {e}")
            return None

    def read_rgbd_pair(self, image_name: str = "head") -> Optional[Dict[str, np.ndarray]]:
        """Read an RGB + depth pair that share the same timestamp.

        Returns:
            Dict with keys: 'rgb', 'depth' or None if not available.
        """
        rgb_shm_name = get_shm_name(image_name, stream="image")
        depth_shm_name = get_shm_name(image_name, stream="depth")

        try:
            if rgb_shm_name not in self.shms:
                self.shms[rgb_shm_name] = shared_memory.SharedMemory(name=rgb_shm_name)
            if depth_shm_name not in self.shms:
                self.shms[depth_shm_name] = shared_memory.SharedMemory(name=depth_shm_name)
        except FileNotFoundError:
            return None

        rgb_shm = self.shms[rgb_shm_name]
        depth_shm = self.shms[depth_shm_name]

        try:
            header_size = ctypes.sizeof(SimpleImageHeader)
            rgb_header = SimpleImageHeader.from_buffer_copy(bytes(rgb_shm.buf[:header_size]))
            depth_header = SimpleImageHeader.from_buffer_copy(bytes(depth_shm.buf[:header_size]))

            # Require matching timestamps for a valid pair
            if rgb_header.timestamp == 0 or depth_header.timestamp == 0:
                return None
            if rgb_header.timestamp != depth_header.timestamp:
                return None

            last_pair_ts = self.last_rgbd_timestamps.get(image_name, 0)
            if rgb_header.timestamp <= last_pair_ts:
                if image_name in self.buffer and image_name in self.depth_buffer:
                    return {"rgb": self.buffer[image_name], "depth": self.depth_buffer[image_name]}
                return None

            # Read RGB
            rgb_payload = bytes(rgb_shm.buf[header_size:header_size + rgb_header.data_size])
            if rgb_header.encoding == 1:
                rgb_encoded = np.frombuffer(rgb_payload, dtype=np.uint8)
                rgb_img = cv2.imdecode(rgb_encoded, cv2.IMREAD_COLOR)
            else:
                rgb_img = np.frombuffer(rgb_payload, dtype=np.uint8)
                expected_size = rgb_header.height * rgb_header.width * rgb_header.channels
                if rgb_img.size != expected_size:
                    return None
                rgb_img = rgb_img.reshape(rgb_header.height, rgb_header.width, rgb_header.channels)

            if rgb_img is None:
                return None

            # Read depth
            depth_payload = bytes(depth_shm.buf[header_size:header_size + depth_header.data_size])
            if depth_header.encoding == 2:
                depth_encoded = np.frombuffer(depth_payload, dtype=np.uint8)
                depth_img = cv2.imdecode(depth_encoded, cv2.IMREAD_UNCHANGED)
            elif depth_header.encoding == 3:
                depth_img = np.frombuffer(depth_payload, dtype=np.uint16)
                expected_size = depth_header.height * depth_header.width
                if depth_img.size != expected_size:
                    return None
                depth_img = depth_img.reshape(depth_header.height, depth_header.width)
            else:
                return None

            if depth_img is None:
                return None

            self.buffer[image_name] = rgb_img
            self.depth_buffer[image_name] = depth_img
            self.last_rgbd_timestamps[image_name] = rgb_header.timestamp
            return {"rgb": rgb_img, "depth": depth_img}

        except Exception as e:
            print(f"[MultiImageReader] Error reading RGBD pair for {image_name}: {e}")
            return None

    def close(self):
        """Close all shared memories"""
        for shm_name, shm in self.shms.items():
            try:
                shm.close()
                print(f"[MultiImageReader] Shared memory closed: {shm_name}")
            except Exception as e:
                print(f"[MultiImageReader] Error closing {shm_name}: {e}")
        self.shms.clear()
        self.buffer.clear()
        self.last_timestamps.clear()


# backward compatible class (single image)
class SharedMemoryWriter:
    """Backward compatible single image writer"""
    
    def __init__(self, shm_name: str = SHM_NAME, shm_size: int = SHM_SIZE):
        self.multi_writer = MultiImageWriter()
    
    def write_image(self, image: np.ndarray) -> bool:
        """Write a single image (as the head image)"""
        return self.multi_writer.write_images({'head': image})
    
    def write_rgbd_pair(self, rgb_image: np.ndarray, depth_image: np.ndarray, *, depth_encoding: str = "png") -> bool:
        """Write a single RGB-D pair (as the head image)"""
        return self.multi_writer.write_rgbd_pair('head', rgb_image, depth_image, depth_encoding=depth_encoding)
    
    def close(self):
        self.multi_writer.close()


class SharedMemoryReader:
    """Backward compatible single image reader"""
    
    def __init__(self, shm_name: str = SHM_NAME):
        self.multi_reader = MultiImageReader()
    
    def read_image(self) -> Optional[np.ndarray]:
        """Read a single image (the head image)"""
        images = self.multi_reader.read_images()
        return images.get('head') if images else None
    
    def read_rgbd_pair(self) -> Optional[Dict[str, np.ndarray]]:
        """Read a single RGB-D pair (the head image)"""
        return self.multi_reader.read_rgbd_pair('head')
    
    def close(self):
        self.multi_reader.close() 

if __name__ == "__main__":
    shm_writer = SharedMemoryWriter()
    shm_reader = SharedMemoryReader()

    while True:
        try:
            shm_writer.write_rgbd_pair(
                rgb_image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                depth_image=np.random.randint(0, 5000, (480, 640), dtype=np.uint16),
                depth_encoding="png"
            )
            img_dic = shm_reader.read_rgbd_pair()
            if img_dic is None:
                print("No RGB-D pair read")
                time.sleep(0.1)
                continue
            rgb_img = img_dic['rgb']
            depth_img = img_dic['depth']
            if rgb_img is not None and depth_img is not None:
                print(f"Read RGB image shape: {rgb_img.shape}, Depth image shape: {depth_img.shape}")
            time.sleep(0.1)
        except KeyboardInterrupt:
            # must close shared memory to avoid resource leak
            shm_writer.close()
            shm_reader.close()
            break
    
