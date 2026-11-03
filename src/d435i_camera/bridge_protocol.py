"""Versioned localhost TCP protocol shared with the ROS2 Foxy bridge."""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from typing import Any, Dict


MAGIC = b"XUK1"
HEADER = struct.Struct("!4sIIII")
MAX_PART_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class BridgePacket:
    metadata: Dict[str, Any]
    color_jpeg: bytes
    depth_zlib: bytes
    annotated_jpeg: bytes


def encode_packet(packet: BridgePacket) -> bytes:
    metadata = json.dumps(packet.metadata, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    parts = (metadata, packet.color_jpeg, packet.depth_zlib, packet.annotated_jpeg)
    if any(len(part) > MAX_PART_BYTES for part in parts):
        raise ValueError("bridge packet part exceeds safety limit")
    return HEADER.pack(MAGIC, *(len(part) for part in parts)) + b"".join(parts)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        data = sock.recv(size - len(chunks))
        if not data:
            raise ConnectionError("bridge socket closed")
        chunks.extend(data)
    return bytes(chunks)


def recv_packet(sock: socket.socket) -> BridgePacket:
    magic, metadata_size, color_size, depth_size, annotated_size = HEADER.unpack(
        recv_exact(sock, HEADER.size)
    )
    if magic != MAGIC:
        raise ValueError("invalid bridge packet magic")
    sizes = (metadata_size, color_size, depth_size, annotated_size)
    if any(size > MAX_PART_BYTES for size in sizes):
        raise ValueError("bridge packet part exceeds safety limit")
    metadata = json.loads(recv_exact(sock, metadata_size).decode("utf-8"))
    return BridgePacket(
        metadata=metadata,
        color_jpeg=recv_exact(sock, color_size),
        depth_zlib=recv_exact(sock, depth_size),
        annotated_jpeg=recv_exact(sock, annotated_size),
    )


def request_json(host: str, port: int, request: Dict[str, Any], timeout_s: float = 5.0) -> Dict[str, Any]:
    payload = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    with socket.create_connection((host, int(port)), timeout=float(timeout_s)) as sock:
        sock.sendall(payload)
        stream = sock.makefile("rb")
        line = stream.readline(MAX_PART_BYTES)
        if not line:
            raise ConnectionError("camera gateway returned no response")
        return json.loads(line.decode("utf-8"))
