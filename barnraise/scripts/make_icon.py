"""Writes scripts/icon-source.png (1024x1024, dependency-free) for `pnpm tauri icon`."""
import struct, zlib
from pathlib import Path

S = 1024
def px(x, y):
    cx, cy = x - S / 2, y - S / 2
    r = (cx * cx + cy * cy) ** 0.5
    if r > S * 0.47:
        return (0, 0, 0, 0)
    roof = y > S * 0.30 + abs(cx) * 0.75 and y < S * 0.44 + abs(cx) * 0.75 and abs(cx) < S * 0.30
    wall = S * 0.48 < y < S * 0.74 and abs(cx) < S * 0.22
    door = S * 0.58 < y < S * 0.74 and abs(cx) < S * 0.07
    if door:
        return (34, 52, 44, 255)
    if roof or wall:
        return (250, 246, 236, 255)
    return (38, 140, 96, 255)
raw = b"".join(b"\x00" + b"".join(struct.pack("BBBB", *px(x, y)) for x in range(S)) for y in range(S))
def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", S, S, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
out = Path(__file__).with_name("icon-source.png")
out.write_bytes(png)
print(out)
