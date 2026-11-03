import socket
import threading

from d435i_camera.bridge_protocol import BridgePacket, encode_packet, recv_packet


def test_bridge_packet_round_trip():
    left, right = socket.socketpair()
    packet = BridgePacket(
        metadata={"schema": "xuk.d435i.bridge/v1", "name": "水瓶"},
        color_jpeg=b"color",
        depth_zlib=b"depth",
        annotated_jpeg=b"annotated",
    )
    sender = threading.Thread(target=lambda: left.sendall(encode_packet(packet)))
    sender.start()
    decoded = recv_packet(right)
    sender.join(timeout=1.0)
    left.close()
    right.close()
    assert decoded == packet
