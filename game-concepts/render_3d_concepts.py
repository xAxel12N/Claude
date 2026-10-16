#!/usr/bin/env python3
"""Render 3D concept frames for five game ideas.

A small NumPy signed-distance-field raymarcher: soft shadows, ambient
occlusion, fog, water reflections, point lights and volumetric glow.

    python3 render_3d_concepts.py            # 1280x720, all scenes
    python3 render_3d_concepts.py --preview  # 480x270, fast
    python3 render_3d_concepts.py tide forest  # only some scenes
"""
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = "/usr/share/fonts/opentype/inter/"


# ---------------------------------------------------------------- math
def c(r, g, b):
    return np.array([[r], [g], [b]], dtype=np.float64)


def dot(a, b):
    return (a * b).sum(0)


def length(a):
    return np.sqrt((a * a).sum(0))


def normalize(a):
    return a / np.maximum(length(a), 1e-9)


def cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def reflect(d, n):
    return d - 2.0 * dot(d, n) * n


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def rep(x, s):
    return x - s * np.round(x / s)


def rot_y(p, a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2]])


def rot_x(p, a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([p[0], ca * p[1] - sa * p[2], sa * p[1] + ca * p[2]])


def rot_z(p, a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([ca * p[0] - sa * p[1], sa * p[0] + ca * p[1], p[2]])


# ---------------------------------------------------------------- noise
def hash2(x, y):
    h = np.sin(x * 127.1 + y * 311.7) * 43758.5453
    return h - np.floor(h)


def hash3(x, y, z):
    h = np.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453
    return h - np.floor(h)


def noise2(x, y):
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    ux, uy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = hash2(ix, iy), hash2(ix + 1, iy)
    cc, d = hash2(ix, iy + 1), hash2(ix + 1, iy + 1)
    return (a + (b - a) * ux) * (1 - uy) + (cc + (d - cc) * ux) * uy


def fbm2(x, y, octaves=5):
    s, a = 0.0, 0.5
    for _ in range(octaves):
        s = s + a * noise2(x, y)
        x, y = 1.6 * x + 1.2 * y, -1.2 * x + 1.6 * y
        a *= 0.5
    return s


def noise3(x, y, z):
    ix, iy, iz = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = x - ix, y - iy, z - iz
    ux, uy, uz = (f * f * (3 - 2 * f) for f in (fx, fy, fz))

    def lerp(a, b, t):
        return a + (b - a) * t

    def h(dx, dy, dz):
        return hash3(ix + dx, iy + dy, iz + dz)

    x00 = lerp(h(0, 0, 0), h(1, 0, 0), ux)
    x10 = lerp(h(0, 1, 0), h(1, 1, 0), ux)
    x01 = lerp(h(0, 0, 1), h(1, 0, 1), ux)
    x11 = lerp(h(0, 1, 1), h(1, 1, 1), ux)
    return lerp(lerp(x00, x10, uy), lerp(x01, x11, uy), uz)


def fbm3(x, y, z, octaves=4):
    s, a = 0.0, 0.5
    for _ in range(octaves):
        s = s + a * noise3(x, y, z)
        x, y, z = 1.7 * x + 0.3, 1.7 * y - 0.5, 1.7 * z + 0.9
        a *= 0.5
    return s


# ---------------------------------------------------------------- SDF primitives
def sd_box(x, y, z, bx, by, bz):
    qx, qy, qz = np.abs(x) - bx, np.abs(y) - by, np.abs(z) - bz
    out = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2)
    return out + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0)


def sd_sphere(p, ctr, r):
    return length(p - ctr) - r


def sd_seg(p, a, b, r):
    pa, ba = p - a, b - a
    h = np.clip(dot(pa, ba) / np.maximum(dot(ba, ba), 1e-9), 0, 1)
    return length(pa - ba * h) - r


def sd_cyl_y(x, y, z, r, h):
    dx = np.sqrt(x * x + z * z) - r
    dy = np.abs(y) - h
    return np.minimum(np.maximum(dx, dy), 0) + np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2)


def sd_capped_cone(x, y, z, h, r1, r2):
    """Cone along y centred at origin, half height h, radius r1 at bottom, r2 at top."""
    qx, qy = np.sqrt(x * x + z * z), y
    k1x, k1y, k2x, k2y = r2, h, r2 - r1, 2 * h
    cax = qx - np.minimum(qx, np.where(qy < 0, r1, r2))
    cay = np.abs(qy) - h
    t = np.clip(((k1x - qx) * k2x + (k1y - qy) * k2y) / (k2x * k2x + k2y * k2y), 0, 1)
    cbx, cby = qx - k1x + k2x * t, qy - k1y + k2y * t
    s = np.where((cbx < 0) & (cay < 0), -1.0, 1.0)
    return s * np.sqrt(np.minimum(cax * cax + cay * cay, cbx * cbx + cby * cby))


def sd_person(p):
    """A standing figure about 1.75 tall, feet at the origin."""
    legs = np.minimum(sd_seg(p, c(-0.1, 0, 0), c(-0.07, 0.85, 0), 0.075),
                      sd_seg(p, c(0.1, 0, 0), c(0.07, 0.85, 0), 0.075))
    torso = sd_seg(p, c(0, 0.92, 0), c(0, 1.38, 0), 0.17)
    head = sd_sphere(p, c(0, 1.6, 0), 0.11)
    arms = np.minimum(sd_seg(p, c(-0.21, 1.36, 0), c(-0.27, 0.95, 0.03), 0.055),
                      sd_seg(p, c(0.21, 1.36, 0), c(0.33, 1.0, 0.08), 0.055))
    pack = sd_box(p[0], p[1] - 1.15, p[2] + 0.2, 0.15, 0.2, 0.08) - 0.03
    return smin(smin(np.minimum(legs, torso), np.minimum(head, arms), 0.05), pack, 0.04)


def union(d, m, d2, mat):
    take = d2 < d
    return np.where(take, d2, d), np.where(take, mat, m)


# materials
CONCRETE, WOOD, CLOTH, LAMP, SAND, MONO, METAL, GOLD, PANEL, BARK, LEAF, GROUND, BEACON, STEEL, ROCK, PAINT = range(16)


# ---------------------------------------------------------------- scenes
class Scene:
    name = ""
    title = ""
    tagline = ""
    far = 200.0
    steps = 160
    step_k = 1.0
    eps = 0.0008
    shadow_max = 40.0
    shadow_k = 12.0
    ao_scale = 1.0
    water = None          # (level, amplitude, freq_x, freq_z, deep colour)
    exposure = 1.0
    lights = ()           # (position (3,1), colour (3,1), casts shadow)
    sun = normalize(c(0, 1, 0))
    sun_col = c(1, 1, 1)
    amb_sky = c(0.1, 0.1, 0.1)
    amb_ground = c(0.03, 0.03, 0.03)

    def bump(self, p, n, m):
        return n

    def volumetrics(self, col, ro, rd, t):
        return col


class DrownedCity(Scene):
    name = "tide"
    title = "HOLLOW TIDE"
    tagline = "Pilot a skiff through a drowned megacity at the end of the world. Salvage, sail, survive the rising sea."
    far = 260.0
    shadow_max = 60.0
    ao_scale = 2.0
    exposure = 1.1
    water = (0.0, 0.05, 0.35, 0.9, c(0.015, 0.03, 0.04))
    sun = normalize(c(0.22, 0.07, 1.0))
    sun_col = c(2.4, 1.25, 0.65)
    amb_sky = c(0.22, 0.17, 0.2)
    amb_ground = c(0.08, 0.05, 0.04)
    lights = ((c(4.3, 2.0, 12.15), c(1.6, 0.8, 0.3), False),)
    cam = (c(4.0, 1.7, 0.0), c(2.6, 5.0, 40.0), 1.5)

    def map(self, p):
        x, y, z = p
        C = 8.0
        cx, cz = np.floor(x / C), np.floor(z / C)
        h1, h2, h3 = hash2(cx, cz), hash2(cx + 17.3, cz - 4.1), hash2(cx - 9.7, cz + 31.3)
        present = (cx != 0) & (h1 > 0.12) & (cz > -2)
        hx, hz = 1.5 + 0.6 * h2, 1.5 + 0.6 * h3
        hy = 4.0 + 19.0 * h1 * h1
        qx = x - ((cx + 0.5) * C + (h3 - 0.5) * 1.0)
        qz = z - ((cz + 0.5) * C + (h2 - 0.5) * 1.0)
        a = h2 * 1.2
        ca, sa = np.cos(a), np.sin(a)
        qx, qz = ca * qx + sa * qz, -sa * qx + ca * qz
        tl = (h1 - 0.5) * 0.1
        ct, st = np.cos(tl), np.sin(tl)
        qx, qy = ct * qx - st * y, st * qx + ct * y
        qy = qy - (hy * 0.5 - 2.0)
        b = sd_box(qx, qy, qz, hx, hy * 0.5, hz)
        # slanted, collapsed roofs on some towers
        slope = np.where(h3 > 0.5, 0.35, -0.35)
        b = np.where(h3 > 0.25, np.maximum(b, (slope * qx + qy - (hy * 0.5 - 1.2)) / np.sqrt(1 + slope ** 2)), b)
        # punched window grid, only in the outer shell
        w = sd_box(rep(qx, 0.9), rep(qy, 1.2), rep(qz, 0.9), 0.27, 0.36, 0.27)
        inner = sd_box(qx, qy, qz, hx - 0.35, hy * 0.5 - 0.3, hz - 0.35)
        b = np.maximum(b, -np.maximum(w, -inner))
        b = np.where(present, b, 1e9)
        lx, lz = x - cx * C, z - cz * C
        bnd = np.minimum(np.minimum(lx, C - lx), np.minimum(lz, C - lz)) + 0.1
        d = np.minimum(b, bnd)
        d = np.where(y > 25, y - 24, d)
        m = np.full(d.shape, CONCRETE)
        # skiff, sailor, lantern pole
        q = rot_y(p - c(4.3, 0.05, 11.2), 0.25)
        hull = sd_box(q[0], q[1], q[2], 0.45, 0.16, 1.35) - 0.1
        hull = np.maximum(hull, -sd_box(q[0], q[1] - 0.25, q[2], 0.36, 0.2, 1.25))
        d, m = union(d, m, hull, WOOD)
        d, m = union(d, m, sd_person(p - c(4.25, 0.12, 10.7)), CLOTH)
        d, m = union(d, m, sd_seg(p, c(4.3, 0.1, 12.2), c(4.3, 2.2, 12.2), 0.03), WOOD)
        d, m = union(d, m, sd_sphere(p, c(4.3, 2.0, 12.15), 0.09), LAMP)
        return d, m

    def material(self, p, n, m):
        x, y, z = p
        N = x.shape[0]
        alb = np.zeros((3, N)) + c(0.4, 0.38, 0.35)
        grime = 0.6 + 0.6 * fbm3(x * 0.7, y * 0.7, z * 0.7, 3)
        streak = 0.75 + 0.25 * noise2(x * 3 + z * 3, y * 0.3)
        alb = alb * grime * streak
        wet = smoothstep(1.6, 0.0, y)
        alb = alb * (1 - wet) + alb * c(0.35, 0.45, 0.38) * wet
        alb = np.where(m == WOOD, c(0.26, 0.15, 0.08), alb)
        alb = np.where(m == CLOTH, c(0.12, 0.16, 0.2), alb)
        spec = np.where(m == CONCRETE, 0.08 + 0.4 * wet, 0.1)
        shin = np.full(N, 20.0)
        emit = np.where(m == LAMP, c(8.0, 4.0, 1.4), 0.0)
        return alb, spec, shin, emit

    def sky(self, ro, rd):
        y = np.clip(rd[1], 0, 1)
        g = y ** 0.45
        col = c(1.05, 0.52, 0.28) * (1 - g) + c(0.1, 0.12, 0.25) * g
        s = np.clip(dot(rd, self.sun), 0, 1)
        col = col + self.sun_col * (0.25 * s ** 6 + 0.5 * s ** 60) + self.sun_col * 6 * (s > 0.9994)
        yy = rd[1] + 0.06
        u, w = rd[0] / yy, rd[2] / yy
        cl = smoothstep(0.5, 0.78, fbm2(u * 0.25 + 3, w * 0.07, 5)) * np.clip(rd[1] * 10, 0, 1)
        clc = c(0.3, 0.17, 0.22) * (1 - s ** 4) + c(1.8, 0.85, 0.45) * s ** 4
        return col * (1 - 0.85 * cl) + clc * 0.85 * cl

    def fog(self, col, t, rd):
        f = 1 - np.exp(-np.minimum(t, 1e4) * 0.011)
        s = np.clip(dot(rd, self.sun), 0, 1)
        fc = c(0.75, 0.42, 0.3) * (1 - s ** 8) + c(1.8, 0.95, 0.5) * s ** 8
        return col * (1 - f) + fc * f


class Monolith(Scene):
    name = "monolith"
    title = "MONOLITH"
    tagline = "Walk an endless desert toward a black slab that wasn't there yesterday. Every night, it moves closer."
    far = 400.0
    steps = 200
    shadow_max = 120.0
    shadow_k = 16.0
    ao_scale = 2.0
    exposure = 0.85
    sun = normalize(c(0.9, 0.32, -0.25))
    sun_col = c(3.2, 2.4, 1.6)
    amb_sky = c(0.25, 0.32, 0.45)
    amb_ground = c(0.25, 0.15, 0.08)

    def __init__(self):
        self.mono_y = float(self.height(np.array([5.0]), np.array([42.0]))[0])
        self.fig_y = float(self.height(np.array([1.2]), np.array([16.0]))[0])
        h0 = float(self.height(np.array([0.0]), np.array([0.0]))[0])
        self.cam = (c(0.0, h0 + 2.2, 0.0), c(3.0, self.mono_y + 4.5, 42.0), 1.6)

    @staticmethod
    def height(x, z):
        n = fbm2(x * 0.012, z * 0.012, 5)
        ridge = 1 - np.abs(np.sin(x * 0.05 + z * 0.03 + n * 6.0))
        return 14.0 * n + 4.0 * ridge * ridge - 9.0

    def map(self, p):
        x, y, z = p
        d = np.where(y > 9, y - 8, (y - self.height(x, z)) * 0.5)
        m = np.full(d.shape, SAND)
        q = rot_z(rot_y(p - c(5.0, self.mono_y + 6.0, 42.0), 0.35), 0.03)
        d, m = union(d, m, sd_box(q[0], q[1], q[2], 1.3, 8.0, 0.32), MONO)
        d, m = union(d, m, sd_person(rot_y(p - c(1.2, self.fig_y - 0.05, 16.0), 0.3)), CLOTH)
        return d, m

    def bump(self, p, n, m):
        x, z = p[0], p[2]
        ph = x * 5.0 + z * 2.0 + 3.0 * noise2(x * 0.5, z * 0.5)
        k = 0.08 * np.cos(ph)
        nb = normalize(n + np.array([k * 0.9, np.zeros_like(k), k * 0.4]))
        return np.where(m == SAND, nb, n)

    def material(self, p, n, m):
        x, y, z = p
        N = x.shape[0]
        v = 0.85 + 0.25 * noise2(x * 0.3, z * 0.3)
        alb = np.zeros((3, N)) + c(0.82, 0.55, 0.33) * v
        alb = np.where(m == MONO, c(0.01, 0.01, 0.012), alb)
        alb = np.where(m == CLOTH, c(0.55, 0.36, 0.22), alb)
        spec = np.where(m == MONO, 1.0, 0.05)
        shin = np.where(m == MONO, 400.0, 8.0)
        return alb, spec, shin, np.zeros((3, N))

    def sky(self, ro, rd):
        y = np.clip(rd[1], 0, 1)
        g = y ** 0.5
        col = c(0.8, 0.7, 0.6) * (1 - g) + c(0.08, 0.22, 0.6) * g
        s = np.clip(dot(rd, self.sun), 0, 1)
        col = col + self.sun_col * (0.05 * s ** 4 + 0.4 * s ** 40) + self.sun_col * 8 * (s > 0.9993)
        return col

    def fog(self, col, t, rd):
        f = 1 - np.exp(-np.minimum(t, 1e4) * 0.0028)
        s = np.clip(dot(rd, self.sun), 0, 1)
        fc = c(0.8, 0.66, 0.52) * (1 - s ** 6) + c(2.0, 1.5, 1.0) * s ** 6
        return col * (1 - f) + fc * f


class OrbitalSalvage(Scene):
    name = "salvage"
    title = "DEAD ORBIT"
    tagline = "Zero-g salvage on a gutted space station. Cut, tether, and haul parts home before your oxygen runs dry."
    far = 60.0
    shadow_max = 30.0
    shadow_k = 40.0
    exposure = 1.0
    sun = normalize(c(-1.0, 0.45, 0.25))
    sun_col = c(3.2, 3.0, 2.8)
    amb_sky = c(0.01, 0.012, 0.02)
    amb_ground = c(0.07, 0.12, 0.22)
    planet_c = c(0.0, -150.0, 70.0)
    planet_r = 135.0
    cam = (c(-1.0, 1.5, 1.5), c(0.5, -1.0, 16.0), 1.3)

    def local(self, p):
        q = p - c(0.0, 0.8, 16.0)
        return rot_z(rot_x(rot_y(q, 0.7), 0.25), 0.15)

    def map(self, p):
        q = self.local(p)
        bound = length(q) - 13.0
        d = np.full(bound.shape, 1e9)
        m = np.full(bound.shape, METAL)
        x, y, z = q
        d, m = union(d, m, sd_cyl_y(y, x, z, 0.35, 9.5), METAL)
        d, m = union(d, m, sd_sphere(q, c(0, 0, 0), 1.4), METAL)
        mod = np.full(d.shape, 1e9)
        for i in (-1, 1):
            mod = np.minimum(mod, sd_cyl_y(x - i * 3.2, z, y, 0.8, 2.6) - 0.05)
        mod = np.maximum(mod, -sd_sphere(q, c(3.2, 0.75, 1.5), 0.8))
        d, m = union(d, m, mod, GOLD)
        for i in (-1, 1):
            sx = x - i * 7.2
            sq = rot_x(np.array([sx, y, z]), 0.5 if i > 0 else -0.2)
            pz = np.abs(sq[2]) - 1.3
            d, m = union(d, m, sd_box(sq[0], sq[1], pz, 2.0, 0.03, 1.15), PANEL)
        for i in range(16):
            ctr = c((hash2(i, 1.3) - 0.5) * 16, (hash2(i, 7.1) - 0.5) * 7, (hash2(i, 3.7) - 0.5) * 10)
            r = rot_x(rot_y(q - ctr, i * 1.7), i * 0.9)
            s = 0.08 + 0.25 * hash2(i, 9.9)
            d, m = union(d, m, sd_box(r[0], r[1], r[2], s, s * 0.4, s * 1.6), METAL)
        d = np.where(bound > 1.0, bound, d)
        return d, m

    def material(self, p, n, m):
        q = self.local(p)
        x, y, z = p
        N = x.shape[0]
        alb = np.zeros((3, N)) + c(0.6, 0.6, 0.62) * (0.7 + 0.4 * noise3(x * 2, y * 2, z * 2))
        foil = 0.55 + 0.7 * fbm3(x * 6, y * 6, z * 6, 3)
        alb = np.where(m == GOLD, c(0.85, 0.6, 0.2) * foil, alb)
        sx = np.abs(q[0]) - 7.2
        grid = (np.abs(rep(sx, 0.5)) < 0.02) | (np.abs(rep(q[2], 0.38)) < 0.02)
        alb = np.where(m == PANEL, np.where(grid, c(0.5, 0.5, 0.55), c(0.03, 0.05, 0.14)), alb)
        spec = np.where(m == GOLD, 1.2, np.where(m == PANEL, 0.9, 0.4))
        shin = np.where(m == GOLD, 30.0, np.where(m == PANEL, 200.0, 60.0))
        return alb, spec, shin, np.zeros((3, N))

    def sky(self, ro, rd):
        N = rd.shape[1]
        col = np.zeros((3, N))
        g = np.floor(rd * 500)
        st = hash3(g[0], g[1], g[2])
        col = col + (st > 0.9985) * (st - 0.9985) / 0.0015 * c(0.9, 0.9, 1.0)
        s = np.clip(dot(rd, self.sun), 0, 1)
        col = col + self.sun_col * (0.02 * s ** 20 + 0.4 * s ** 400) + self.sun_col * 20 * (s > 0.9997)
        oc = ro - self.planet_c
        b = dot(oc, rd)
        cc = dot(oc, oc) - self.planet_r ** 2
        disc = b * b - cc
        hit = (disc > 0) & (-b - np.sqrt(np.maximum(disc, 0)) > 0)
        # atmosphere glow around the limb
        closest = np.sqrt(np.maximum(dot(oc, oc) - b * b, 0)) - self.planet_r
        cp = normalize(ro + rd * np.maximum(-b, 0) - self.planet_c)
        lit = np.clip(dot(cp, self.sun) + 0.3, 0, 1)
        glow = np.exp(-np.maximum(closest, 0) / 3.0) * lit * (b < 0)
        col = col + c(0.35, 0.6, 1.3) * glow * 1.4 * (~hit)
        if hit.any():
            idx = np.nonzero(hit)[0]
            r = rd[:, idx]
            t = -b[idx] - np.sqrt(disc[idx])
            nn = normalize(ro[:, idx] + r * t - self.planet_c)
            w = fbm3(nn[0] * 3, nn[1] * 3, nn[2] * 3, 4)
            land = fbm3(nn[0] * 8 + w, nn[1] * 8 + 2, nn[2] * 8 - w, 6)
            landm = smoothstep(0.52, 0.56, land)
            lc = c(0.12, 0.17, 0.06) * (1 - w) + c(0.38, 0.3, 0.18) * w
            alb = c(0.01, 0.035, 0.11) * (1 - landm) + lc * landm
            cw = fbm3(nn[0] * 6 + 7, nn[1] * 6 + 1, nn[2] * 6 + 3, 3)
            clouds = smoothstep(0.5, 0.78, fbm3(nn[0] * 16 + cw * 2, nn[1] * 16, nn[2] * 16 - cw * 2, 6))
            alb = alb * (1 - clouds) + c(0.95, 0.95, 0.97) * clouds
            dif = np.clip(dot(nn, self.sun), 0, 1)
            pc = alb * self.sun_col * dif * 0.9
            glint = np.clip(dot(reflect(r, nn), self.sun), 0, 1) ** 80 * (1 - landm) * (1 - clouds)
            pc = pc + self.sun_col * glint * 0.5 * dif
            gg = np.floor(nn * 600)
            city = (hash3(gg[0], gg[1], gg[2]) > 0.96) * landm * (1 - clouds)
            night = smoothstep(0.05, -0.15, dot(nn, self.sun))
            pc = pc + c(1.2, 0.6, 0.2) * city * night * 0.6
            mu = np.clip(dot(nn, -r), 0, 1)
            rim = (1 - mu) ** 2.5
            pc = pc * (1 - rim * 0.6) + c(0.35, 0.6, 1.3) * rim * (dif + 0.05) * 1.2
            col[:, idx] = pc
        return col

    def fog(self, col, t, rd):
        return col


class SignalPine(Scene):
    name = "forest"
    title = "SIGNAL PINE"
    tagline = "A dead radio tower in a foggy forest started broadcasting again. Go find out who's transmitting."
    far = 90.0
    steps = 180
    step_k = 0.75
    shadow_max = 25.0
    ao_scale = 1.0
    exposure = 1.3
    sun = normalize(c(-0.25, 0.45, 1.0))
    sun_col = c(0.22, 0.28, 0.42)
    amb_sky = c(0.03, 0.04, 0.065)
    amb_ground = c(0.01, 0.01, 0.012)

    def __init__(self):
        self.fig_y = float(self.ground(np.array([0.5]), np.array([10.0]))[0])
        self.lamp = c(0.86, self.fig_y + 1.0, 10.12)
        self.beacon = c(0.0, 30.4, 42.0)
        self.lights = ((self.lamp, c(1.8, 0.95, 0.35), True), (self.beacon, c(7.0, 0.25, 0.15), False))
        self.cam = (c(0.3, self.fig_y + 1.8, 0.0), c(0.0, 9.0, 42.0), 1.45)

    @staticmethod
    def ground(x, z):
        return 0.4 * fbm2(x * 0.15, z * 0.15, 3) + 0.03 * np.abs(x)

    def map(self, p):
        x, y, z = p
        d = y - self.ground(x, z)
        m = np.full(d.shape, GROUND)
        C = 5.0
        cx, cz = np.floor(x / C), np.floor(z / C)
        h1, h2, h3 = hash2(cx, cz), hash2(cx + 5.1, cz + 2.3), hash2(cx - 3.3, cz + 8.8)
        tx = (cx + 0.5) * C + (h2 - 0.5) * 0.8
        tz = (cz + 0.5) * C + (h3 - 0.5) * 0.8
        skip = ((np.abs(tx) < 3.0) & (tz < 36)) | ((np.abs(tx) < 7) & (np.abs(tz - 42) < 7)) | (h1 < 0.08)
        idx = np.nonzero(~skip & (y < 15))[0]
        if idx.size:
            s = 0.85 + 0.35 * h1[idx]
            qx, qy, qz = x[idx] - tx[idx], y[idx], z[idx] - tz[idx]
            trunk = sd_cyl_y(qx, qy - 4.0 * s, qz, 0.12 + 0.08 * h3[idx], 4.0 * s)
            leaves = np.full(idx.shape, 1e9)
            for k in range(4):
                hh = 1.2 * s
                by = (2.2 + 1.5 * k) * s + hh
                leaves = np.minimum(leaves, sd_capped_cone(qx, qy - by, qz, hh, (1.6 - 0.3 * k) * s, 0.05))
            leaves = leaves + 0.14 * noise3(x[idx] * 3, y[idx] * 3, z[idx] * 3) - 0.07
            dd, mm = union(d[idx], m[idx], trunk, BARK)
            dd, mm = union(dd, mm, leaves, LEAF)
            d[idx], m[idx] = dd, mm
        lx, lz = x - cx * C, z - cz * C
        bnd = np.minimum(np.minimum(lx, C - lx), np.minimum(lz, C - lz)) + 0.1
        d = np.where(y < 15, np.minimum(d, bnd), d)
        # lattice radio tower
        tq = p - c(0.0, 0.0, 42.0)
        tb = sd_box(tq[0], tq[1] - 16.5, tq[2], 2.6, 17.0, 2.6)
        idx = np.nonzero(tb < d)[0]
        if idx.size:
            q = tq[:, idx]
            tw = np.full(idx.shape, 1e9)
            corners = ((1, 1), (1, -1), (-1, -1), (-1, 1))
            for sx, sz in corners:
                tw = np.minimum(tw, sd_seg(q, c(sx * 2.2, 0, sz * 2.2), c(sx * 0.25, 30, sz * 0.25), 0.07))
            k = np.clip(np.floor((q[1] - 1.5) / 3.0), 0, 8)
            y0 = 1.5 + 3 * k
            y1 = y0 + 3
            w0, w1 = 2.2 - 1.95 * y0 / 30, 2.2 - 1.95 * y1 / 30
            for i in range(4):
                ax, az = corners[i]
                bx, bz = corners[(i + 1) % 4]
                a0 = np.array([ax * w0, y0, az * w0])
                b0 = np.array([bx * w0, y0, bz * w0])
                a1 = np.array([ax * w1, y1, az * w1])
                b1 = np.array([bx * w1, y1, bz * w1])
                tw = np.minimum(tw, sd_seg(q, a0, b0, 0.04))
                tw = np.minimum(tw, sd_seg(q, a1, b1, 0.04))
                tw = np.minimum(tw, sd_seg(q, a0, b1, 0.03))
                tw = np.minimum(tw, sd_seg(q, b0, a1, 0.03))
            tw = np.minimum(tw, sd_seg(q, c(0, 30, 0), c(0, 34, 0), 0.05))
            dd, mm = union(d[idx], m[idx], tw, STEEL)
            dd, mm = union(dd, mm, sd_sphere(q, c(0, 30.4, 0), 0.28), BEACON)
            d[idx], m[idx] = dd, mm
        d, m = union(d, m, sd_person(p - c(0.5, self.fig_y, 10.0)), CLOTH)
        d, m = union(d, m, sd_sphere(p, self.lamp, 0.07), LAMP)
        return d, m

    def material(self, p, n, m):
        x, y, z = p
        N = x.shape[0]
        v = 0.7 + 0.6 * noise2(x * 1.5, z * 1.5)
        path = smoothstep(1.6, 0.6, np.abs(x - 0.3 * np.sin(z * 0.15)))
        alb = np.zeros((3, N)) + (c(0.06, 0.07, 0.04) * (1 - path) + c(0.16, 0.13, 0.1) * path) * v
        alb = np.where(m == BARK, c(0.12, 0.08, 0.06), alb)
        alb = np.where(m == LEAF, c(0.04, 0.08, 0.05) * (0.6 + 0.8 * noise3(x * 4, y * 4, z * 4)), alb)
        alb = np.where(m == STEEL, c(0.25, 0.22, 0.2), alb)
        alb = np.where(m == CLOTH, c(0.3, 0.12, 0.08), alb)
        spec = np.where(m == STEEL, 0.5, 0.05)
        shin = np.full(N, 20.0)
        emit = np.where(m == LAMP, c(12.0, 6.0, 2.0), np.where(m == BEACON, c(30.0, 1.0, 0.5), 0.0))
        return alb, spec, shin, emit

    def sky(self, ro, rd):
        y = np.clip(rd[1], 0, 1)
        col = c(0.06, 0.07, 0.1) * (1 - y) + c(0.01, 0.015, 0.03) * y
        s = np.clip(dot(rd, self.sun), 0, 1)
        return col + c(0.5, 0.55, 0.7) * (0.1 * s ** 30) + c(1.5, 1.6, 1.8) * (s > 0.9996)

    def fog(self, col, t, rd):
        f = 1 - np.exp(-np.minimum(t, 1e4) * 0.045)
        s = np.clip(dot(rd, self.sun), 0, 1)
        fc = c(0.045, 0.055, 0.08) + c(0.12, 0.14, 0.2) * s ** 6
        return col * (1 - f) + fc * f

    def volumetrics(self, col, ro, rd, t):
        T = np.minimum(t, self.far)
        for pos, lc, _ in self.lights:
            oc = pos - ro
            h = dot(oc, rd)
            b = np.sqrt(np.maximum(dot(oc, oc) - h * h, 1e-4))
            integ = (np.arctan((T - h) / b) - np.arctan(-h / b)) / b
            col = col + lc * integ * 0.035 * np.exp(-np.maximum(h, 0) * 0.02)
        return col


class LastLight(Scene):
    name = "lighthouse"
    title = "LAST LIGHT"
    tagline = "Keep the last lighthouse burning through an endless storm. Guide lost ships in, keep the dark things out."
    far = 250.0
    steps = 180
    step_k = 0.7
    shadow_max = 40.0
    ao_scale = 1.5
    exposure = 1.25
    water = (0.0, 0.3, 0.22, 0.5, c(0.01, 0.025, 0.035))
    sun = normalize(c(-0.85, 0.1, 0.55))
    sun_col = c(1.1, 0.7, 0.5)
    amb_sky = c(0.12, 0.16, 0.24)
    amb_ground = c(0.02, 0.03, 0.04)
    lamp = c(6.0, 17.1, 45.0)
    lights = ((c(6.0, 17.1, 45.0), c(18.0, 14.0, 8.0), True),)
    beam = normalize(c(-0.88, -0.04, -0.48))
    cam = (c(-4.0, 2.0, 14.0), c(4.5, 9.5, 45.0), 1.45)

    def map(self, p):
        x, y, z = p
        isl = sd_box((x - 6) / 1.0, y + 3.0, (z - 45) / 1.0, 11.0, 6.0, 8.0) - 2.0
        near = np.nonzero(isl < 6.0)[0]
        if near.size:
            xs, ys, zs = x[near], y[near], z[near]
            e = np.sqrt(((xs - 6) / 13) ** 2 + ((ys + 2.5) / 6.5) ** 2 + ((zs - 45) / 10) ** 2) - 1.0
            e = e * 6.0 + 2.6 * fbm3(xs * 0.35, ys * 0.35, zs * 0.35, 5) - 1.3
            isl[near] = e
        d = isl
        m = np.full(d.shape, ROCK)
        rocks = np.minimum(sd_sphere(p, c(-6.0, -0.6, 32.0), 1.6), sd_sphere(p, c(17.0, -0.8, 40.0), 2.2))
        rocks = rocks + 0.6 * noise3(x * 1.2, y * 1.2, z * 1.2) - 0.3
        d, m = union(d, m, rocks, ROCK)
        q = p - c(6.0, 0.0, 45.0)
        qx, qy, qz = q
        tower = sd_capped_cone(qx, qy - 9.0, qz, 6.5, 1.35, 0.9)
        d, m = union(d, m, tower, PAINT)
        gallery = sd_cyl_y(qx, qy - 15.7, qz, 1.3, 0.12)
        rail = sd_cyl_y(qx, qy - 16.2, qz, 1.28, 0.42)
        rail = np.maximum(rail, -sd_cyl_y(qx, qy - 16.2, qz, 1.24, 0.6))
        rail = np.maximum(rail, np.abs(rep(np.arctan2(qx, qz) * 1.3, 0.25)) - 0.03)
        d, m = union(d, m, np.minimum(gallery, sd_cyl_y(qx, qy - 16.6, qz, 1.3, 0.03)), STEEL)
        d, m = union(d, m, rail, STEEL)
        d, m = union(d, m, sd_cyl_y(qx, qy - 17.1, qz, 0.68, 0.6), LAMP)
        d, m = union(d, m, sd_capped_cone(qx, qy - 18.2, qz, 0.5, 0.85, 0.08), STEEL)
        house = sd_box(qx - 2.6, qy - 3.6, qz + 1.0, 1.4, 1.1, 1.1)
        roof = sd_capped_cone(qx - 2.6, qy - 5.2, qz + 1.0, 0.5, 1.7, 0.2)
        d, m = union(d, m, np.minimum(house, roof), PAINT)
        return d, m

    def material(self, p, n, m):
        x, y, z = p
        N = x.shape[0]
        v = 0.5 + 0.7 * fbm3(x * 0.8, y * 0.8, z * 0.8, 4)
        alb = np.zeros((3, N)) + c(0.18, 0.17, 0.16) * v
        moss = smoothstep(0.55, 0.9, n[1]) * smoothstep(1.0, 2.5, y)
        alb = alb * (1 - moss) + c(0.06, 0.1, 0.04) * moss
        stripe = np.floor((y - 2.5) / 2.6) % 2 == 0
        tower = np.where(stripe, c(0.85, 0.82, 0.78), c(0.55, 0.06, 0.05))
        house = (np.abs(x - 8.6) < 1.7) & (y < 6.0)
        alb = np.where(m == PAINT, np.where(house, c(0.7, 0.68, 0.64), tower), alb)
        alb = np.where(m == STEEL, c(0.08, 0.08, 0.08), alb)
        spec = np.where(m == ROCK, 0.25, 0.3)
        shin = np.full(N, 30.0)
        emit = np.where(m == LAMP, c(30.0, 24.0, 14.0), 0.0)
        win = (m == PAINT) & house & (np.abs(y - 3.7) < 0.35) & (np.abs(rep(z - 44.0, 0.9)) < 0.22) & (n[0] < -0.5)
        emit = np.where(win, c(4.0, 2.2, 0.8), emit)
        return alb, spec, shin, emit

    def sky(self, ro, rd):
        y = np.clip(rd[1], 0, 1)
        s = np.clip(dot(rd, self.sun), 0, 1)
        col = c(0.32, 0.36, 0.42) * (1 - y ** 0.5) + c(0.05, 0.07, 0.11) * y ** 0.5
        col = col + c(1.0, 0.55, 0.3) * s ** 6 * 0.7
        yy = rd[1] + 0.1
        u, w = rd[0] / yy, rd[2] / yy
        n = fbm2(u * 0.6, w * 0.6, 6)
        cl = smoothstep(0.3, 0.62, n) * np.clip(rd[1] * 12, 0, 1)
        clc = c(0.06, 0.07, 0.09) * (1 - s ** 3) + c(0.5, 0.3, 0.22) * s ** 3
        return col * (1 - cl * 0.9) + clc * cl * 0.9

    def fog(self, col, t, rd):
        f = 1 - np.exp(-np.minimum(t, 1e4) * 0.006)
        s = np.clip(dot(rd, self.sun), 0, 1)
        fc = c(0.2, 0.24, 0.3) + c(0.6, 0.35, 0.2) * s ** 6
        return col * (1 - f) + fc * f

    def volumetrics(self, col, ro, rd, t):
        T = np.minimum(t, 160.0)
        acc = np.zeros(rd.shape[1])
        n = 64
        for i in range(n):
            s = (i + 0.5) / n * T
            pp = ro + rd * s
            v = pp - self.lamp
            for direction in (self.beam, -self.beam):
                along = dot(v, direction)
                perp = length(v - direction * along)
                width = 0.45 + np.maximum(along, 0) * 0.09
                acc = acc + (along > 0) * np.exp(-(perp / width) ** 2 * 2.0) * np.exp(-along * 0.012) / width
            acc = acc + np.exp(-length(v) * 0.6) * 0.5
        col = col + c(1.0, 0.85, 0.6) * acc * (T / n) * 0.12
        return col


SCENES = [DrownedCity, Monolith, OrbitalSalvage, SignalPine, LastLight]


# ---------------------------------------------------------------- renderer
def march(scene, ro, rd, tmax, steps=None):
    n = rd.shape[1]
    t = np.full(n, 0.02)
    hit = np.zeros(n, bool)
    alive = np.ones(n, bool)
    for _ in range(steps or scene.steps):
        ia = np.nonzero(alive)[0]
        if ia.size == 0:
            break
        p = ro[:, ia] + rd[:, ia] * t[ia]
        d, _ = scene.map(p)
        h = d < scene.eps * t[ia]
        hit[ia[h]] = True
        t[ia] += np.where(h, 0, d * scene.step_k)
        stop = h | (t[ia] > tmax[ia])
        alive[ia[stop]] = False
    return t, hit


def calc_normal(scene, p):
    e = 0.002
    n = np.zeros_like(p)
    for k in ((1, -1, -1), (-1, -1, 1), (-1, 1, -1), (1, 1, 1)):
        kk = c(*k)
        d, _ = scene.map(p + kk * e)
        n = n + kk * d
    return normalize(n)


def soft_shadow(scene, p, ldir, tmax, steps=56):
    n = p.shape[1]
    res = np.ones(n)
    t = np.full(n, 0.05)
    alive = np.ones(n, bool)
    if ldir.shape[1] == 1:
        ldir = np.broadcast_to(ldir, p.shape)
    tmax = np.broadcast_to(tmax, (n,))
    for _ in range(steps):
        ia = np.nonzero(alive)[0]
        if ia.size == 0:
            break
        d, _ = scene.map(p[:, ia] + ldir[:, ia] * t[ia])
        res[ia] = np.minimum(res[ia], scene.shadow_k * d / t[ia])
        t[ia] += np.clip(d, 0.04, scene.shadow_max / 20)
        alive[ia] = (res[ia] > 0.002) & (t[ia] < tmax[ia])
    return np.clip(res, 0, 1)


def ambient_occlusion(scene, p, n):
    occ, w = 0.0, 1.0
    for i in range(5):
        h = scene.ao_scale * (0.03 + 0.12 * i)
        d, _ = scene.map(p + n * h)
        occ = occ + (h - d) * w
        w *= 0.75
    return np.clip(1 - 2.0 * occ / scene.ao_scale, 0, 1)


def shade(scene, ro, rd, t, depth):
    p = ro + rd * t
    _, m = scene.map(p)
    n = scene.bump(p, calc_normal(scene, p), m)
    alb, spec, shin, emit = scene.material(p, n, m)
    pos = p + n * 0.01
    dif = np.clip(dot(n, scene.sun), 0, 1)
    sh = np.zeros_like(dif)
    lit = np.nonzero(dif > 0)[0]
    if lit.size:
        sh[lit] = soft_shadow(scene, pos[:, lit], scene.sun, scene.shadow_max)
    occ = ambient_occlusion(scene, p, n)
    hv = normalize(scene.sun - rd)
    fre = 0.04 + 0.96 * (1 - np.clip(dot(n, -rd), 0, 1)) ** 5
    spe = np.clip(dot(n, hv), 0, 1) ** shin * dif * sh
    col = alb * (scene.sun_col * dif * sh
                 + scene.amb_sky * (0.5 + 0.5 * n[1]) * occ
                 + scene.amb_ground * (0.5 - 0.5 * n[1]) * occ)
    col = col + scene.sun_col * spe * spec * (0.2 + fre)
    for lpos, lcol, shadows in scene.lights:
        lv = lpos - pos
        dist = length(lv)
        ld = lv / dist
        ldif = np.clip(dot(n, ld), 0, 1)
        lsh = np.ones_like(ldif)
        if shadows:
            idx = np.nonzero((ldif > 0) & (dist < 40))[0]
            lsh = np.zeros_like(ldif)
            if idx.size:
                lsh[idx] = soft_shadow(scene, pos[:, idx], ld[:, idx], dist[idx] - 0.2, steps=40)
        col = col + alb * lcol * ldif * lsh / (dist * dist + 0.3)
    if depth == 0:
        gl = np.nonzero(spec > 0.85)[0]
        if gl.size:
            r = reflect(rd[:, gl], n[:, gl])
            rc = render(scene, pos[:, gl], r, depth + 1)
            col[:, gl] = col[:, gl] + rc * (fre[gl] * 0.9 + 0.08) * spec[gl] * occ[gl]
    return col + emit


def render(scene, ro, rd, depth=0):
    n = rd.shape[1]
    if ro.shape[1] == 1:
        ro = np.broadcast_to(ro, rd.shape).copy()
    tmax = np.full(n, scene.far)
    tw = np.full(n, np.inf)
    if scene.water is not None and depth == 0:
        level = scene.water[0]
        down = rd[1] < -1e-4
        tw[down] = (level - ro[1, down]) / rd[1, down]
        tmax = np.minimum(tmax, np.where(tw > 0, tw, np.inf))
    t, hit = march(scene, ro, rd, tmax)
    col = scene.sky(ro, rd)
    if hit.any():
        idx = np.nonzero(hit)[0]
        col[:, idx] = shade(scene, ro[:, idx], rd[:, idx], t[idx], depth)
    water = ~hit & np.isfinite(tw) & (tw < scene.far)
    if water.any():
        idx = np.nonzero(water)[0]
        level, amp, fx, fz, deep = scene.water
        wr = rd[:, idx]
        wp = ro[:, idx] + wr * tw[idx]
        def h(xx, zz):
            return amp * fbm2(xx * fx, zz * fz, 5)
        e = 0.04
        h0 = h(wp[0], wp[2])
        wn = normalize(np.array([-(h(wp[0] + e, wp[2]) - h0) / e,
                                 np.ones_like(h0),
                                 -(h(wp[0], wp[2] + e) - h0) / e]))
        rr = reflect(wr, wn)
        rr[1] = np.abs(rr[1]) + 0.002
        refl = render(scene, wp + c(0, 0.02, 0), rr, depth + 1)
        fre = 0.02 + 0.98 * (1 - np.clip(dot(wn, -wr), 0, 1)) ** 5
        col[:, idx] = deep * (1 - fre) + refl * fre
    tt = np.where(hit, t, np.where(water, tw, 1e5))
    # open sky only takes haze near the horizon, so clouds and gradients stay visible
    sky_t = 300.0 * (1 - smoothstep(0.0, 0.35, rd[1]))
    col = scene.fog(col, np.where(hit | water, tt, sky_t), rd)
    col = scene.volumetrics(col, ro, rd, tt)
    return col


def aces(x):
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def render_scene(scene, width, height):
    ro, ta, focal = scene.cam
    fw = normalize(ta - ro)
    rt = normalize(cross(c(0, 1, 0), fw))
    up = cross(fw, rt)
    jj, ii = np.mgrid[0:height, 0:width]
    px = ((2 * (ii + 0.5) - width) / height).ravel()
    py = (-(2 * (jj + 0.5) - height) / height).ravel()
    rd = normalize(rt * px + up * py + fw * focal)
    col = np.zeros((3, rd.shape[1]))
    chunk = 120000
    for s in range(0, rd.shape[1], chunk):
        col[:, s:s + chunk] = render(scene, ro, rd[:, s:s + chunk])
    col = aces(col * scene.exposure)
    col = col ** (1 / 2.2)
    vig = 1 - 0.32 * (px * px * 0.6 + py * py)
    grain = (hash2(px * 91.7, py * 53.3) - 0.5) * 0.025
    col = np.clip(col * vig + grain, 0, 1)
    img = (col.T.reshape(height, width, 3) * 255).astype(np.uint8)
    return Image.fromarray(img)


def add_title(img, scene, index):
    w, h = img.size
    s = w / 1280
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    band = int(170 * s)
    for i in range(band):
        a = int(200 * (i / band) ** 1.6)
        d.line([(0, h - band + i), (w, h - band + i)], fill=(0, 0, 0, a))
    img = Image.alpha_composite(img.convert("RGBA"), overlay)
    d = ImageDraw.Draw(img)
    big = ImageFont.truetype(FONT_DIR + "InterDisplay-Black.otf", int(54 * s))
    small = ImageFont.truetype(FONT_DIR + "Inter-Regular.otf", int(19 * s))
    tiny = ImageFont.truetype(FONT_DIR + "Inter-SemiBold.otf", int(13 * s))
    x0 = int(48 * s)
    d.text((x0, h - int(128 * s)), f"CONCEPT 0{index} / 05", font=tiny, fill=(255, 255, 255, 150))
    d.text((x0, h - int(110 * s)), scene.title, font=big, fill=(255, 255, 255, 245))
    d.text((x0, h - int(44 * s)), scene.tagline, font=small, fill=(230, 230, 230, 220))
    return img.convert("RGB")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    width, height = (480, 270) if "--preview" in sys.argv else (1280, 720)
    for i, cls in enumerate(SCENES, 1):
        if args and cls.name not in args:
            continue
        scene = cls()
        start = time.time()
        img = render_scene(scene, width, height)
        img = add_title(img, scene, i)
        out = f"3d_{i:02d}_{scene.name}.png"
        img.save(out)
        print(f"{out}: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
