"""Python 3.8-compatible client for the Python 3.10 camera gateway."""

import json
import socket
import struct


MAGIC = b"XUK1"
HEADER = struct.Struct("!4sIIII")
MAX_PART_BYTES = 128 * 1024 * 1024


def recv_exact(sock, size):
    chunks = bytearray()
    while len(chunks) < size:
        data = sock.recv(size - len(chunks))
        if not data:
            raise ConnectionError("camera gateway stream closed")
        chunks.extend(data)
    return bytes(chunks)


def recv_packet(sock):
    fields = HEADER.unpack(recv_exact(sock, HEADER.size))
    if fields[0] != MAGIC:
        raise ValueError("invalid camera gateway packet magic")
    sizes = fields[1:]
    if any(size > MAX_PART_BYTES for size in sizes):
        raise ValueError("camera gateway packet exceeds safety limit")
    metadata = json.loads(recv_exact(sock, sizes[0]).decode("utf-8"))
    return {
        "metadata": metadata,
        "color_jpeg": recv_exact(sock, sizes[1]),
        "depth_zlib": recv_exact(sock, sizes[2]),
        "annotated_jpeg": recv_exact(sock, sizes[3]),
    }


def request_json(host, port, request, timeout_s=8.0):
    payload = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with socket.create_connection((host, int(port)), timeout=float(timeout_s)) as sock:
        sock.sendall(payload + b"\n")
        line = sock.makefile("rb").readline(MAX_PART_BYTES)
        if not line:
            raise ConnectionError("camera gateway returned no response")
        return json.loads(line.decode("utf-8"))
