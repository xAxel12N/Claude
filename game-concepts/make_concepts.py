"""Render pixel-art concept mockups for the game ideas (320x180, upscaled 4x)."""
import math, random
from PIL import Image, ImageDraw, ImageFont

W, H, SCALE = 320, 180, 4
FONT = ImageFont.load_default()

def save(img, name):
    img.resize((W * SCALE, H * SCALE), Image.NEAREST).save(name)

def text(d, xy, s, fill):
    d.text(xy, s, font=FONT, fill=fill)

# ---------- 1. LANTERN BELOW: light is your health ----------
def lantern_below():
    random.seed(1)
    img = Image.new("RGB", (W, H), (8, 6, 12))
    px = img.load()
    T = 10
    cols, rows = W // T, H // T
    solid = [[False] * cols for _ in range(rows)]
    for y in range(rows):
        for x in range(cols):
            # carve a winding tunnel
            cy = 9 + 4 * math.sin(x * 0.35) + 2 * math.sin(x * 0.9)
            solid[y][x] = abs(y - cy) > 3 + random.random() * 1.5
    ores = {(x, y) for y in range(rows) for x in range(cols)
            if solid[y][x] and random.random() < 0.06}
    lx, ly = 120, 92  # lantern position
    for y in range(H):
        for x in range(W):
            tx, ty = x // T, y // T
            dist = math.hypot(x - lx, y - ly)
            light = max(0.0, 1 - dist / 140) ** 1.1
            light = max(0.07, round(light * 6) / 6)  # banded, retro falloff
            if solid[ty][tx]:
                n = ((x * 7 + y * 13) ^ (tx * 31)) % 9
                base = (70 + n * 3, 52 + n * 2, 44 + n)
                if (tx, ty) in ores and 3 < x % T < 7 and 3 < y % T < 7:
                    base = (90, 230, 255)
                if x % T == 0 or y % T == 0:
                    base = tuple(c // 2 for c in base)
            else:
                base = (28, 22, 30)
            warm = (1.0, 0.85, 0.6)
            px[x, y] = tuple(int(c * light * w) for c, w in zip(base, warm))
            if (tx, ty) in ores and solid[ty][tx] and 4 <= x % T <= 5 and 4 <= y % T <= 5 and light <= 0.07:
                px[x, y] = (30, 70, 90)  # faint glints in the dark
    d = ImageDraw.Draw(img)
    # miner
    d.rectangle([lx - 9, ly - 2, lx - 4, ly + 7], fill=(200, 80, 60))
    d.rectangle([lx - 9, ly - 7, lx - 4, ly - 3], fill=(240, 200, 160))
    d.rectangle([lx - 10, ly - 9, lx - 3, ly - 7], fill=(230, 190, 40))
    d.line([lx - 3, ly, lx, ly - 3], fill=(150, 120, 90))
    d.ellipse([lx - 1, ly - 3, lx + 3, ly + 1], fill=(255, 230, 120))
    # creature eyes lurking at edge of light
    for ex, ey in [(250, 70), (262, 112), (30, 60)]:
        d.point([(ex, ey), (ex + 4, ey)], fill=(255, 40, 40))
    # HUD
    d.rectangle([6, 6, 86, 14], outline=(120, 100, 70))
    d.rectangle([8, 8, 60, 12], fill=(255, 190, 70))
    text(d, (90, 4), "OIL", (255, 190, 70))
    text(d, (6, 18), "DEPTH 142m", (180, 170, 160))
    text(d, (244, 4), "x3 ORE", (90, 230, 255))
    text(d, (6, H - 14), "LANTERN BELOW - concept", (90, 80, 70))
    save(img, "01_lantern_below.png")

# ---------- 2. TINY ORBIT: one-touch gravity slingshot puzzles ----------
def tiny_orbit():
    random.seed(2)
    img = Image.new("RGB", (W, H), (10, 8, 30))
    d = ImageDraw.Draw(img)
    for _ in range(140):
        x, y = random.randrange(W), random.randrange(H)
        b = random.choice([60, 90, 140, 220])
        d.point((x, y), fill=(b, b, min(255, b + 30)))
    planets = [(110, 70, 18, (255, 120, 90)), (210, 120, 12, (120, 220, 160)), (250, 45, 8, (190, 140, 255))]
    for x, y, r, c in planets:
        for k in (2.2, 3.0):  # gravity wells
            rr = int(r * k)
            for a in range(0, 360, 8):
                t = math.radians(a)
                d.point((x + rr * math.cos(t), y + rr * math.sin(t)), fill=tuple(v // 3 for v in c))
        d.ellipse([x - r, y - r, x + r, y + r], fill=c)
        d.ellipse([x - r + 3, y - r + 3, x - r // 3, y - r // 3], fill=tuple(min(255, v + 50) for v in c))
        d.chord([x - r, y - r, x + r, y + r], 20, 160, fill=tuple(int(v * 0.6) for v in c))
    # planned slingshot path (quadratic beziers): ship -> around red planet -> past green -> portal
    def bez(p0, p1, p2, n=60):
        return [((1-t)**2*p0[0] + 2*(1-t)*t*p1[0] + t*t*p2[0],
                 (1-t)**2*p0[1] + 2*(1-t)*t*p1[1] + t*t*p2[1]) for t in (i / n for i in range(n))]
    pts = bez((24, 150), (60, 60), (100, 44)) + bez((100, 44), (150, 40), (150, 85)) \
        + bez((150, 85), (160, 160), (240, 150)) + bez((240, 150), (270, 148), (290, 150))
    for i, p in enumerate(pts):
        if i % 3 == 0:
            d.point(p, fill=(255, 230, 90) if i < 120 else (150, 140, 90))
    # ship at start, goal portal
    d.polygon([(20, 146), (26, 152), (16, 154)], fill=(255, 230, 90))
    gx, gy = 290, 150
    for r, c in [(9, (60, 200, 255)), (6, (20, 40, 90)), (3, (140, 240, 255))]:
        d.ellipse([gx - r, gy - r, gx + r, gy + r], outline=c)
    text(d, (6, 4), "LEVEL 3-7", (200, 200, 255))
    text(d, (240, 4), "FUEL 1/1", (255, 230, 90))
    text(d, (6, H - 14), "TINY ORBIT - concept", (90, 90, 140))
    save(img, "02_tiny_orbit.png")

# ---------- 3. DAEMON KEEPER: tower defense inside a Linux box ----------
def daemon_keeper():
    random.seed(3)
    img = Image.new("RGB", (W, H), (4, 10, 6))
    d = ImageDraw.Draw(img)
    G, DIM = (80, 255, 120), (20, 60, 30)
    for x in range(0, W, 10):
        for y in range(20, H - 20, 10):
            d.point((x, y), fill=DIM)
    path = [(0, 50), (90, 50), (90, 130), (190, 130), (190, 70), (270, 70), (270, 140), (320, 140)]
    for a, b in zip(path, path[1:]):
        d.line([a, b], fill=(30, 90, 45), width=9)
        d.line([a, b], fill=(10, 30, 15), width=5)
    # enemies = rogue processes
    def along(t):
        segs = list(zip(path, path[1:]))
        lens = [math.dist(a, b) for a, b in segs]
        t *= sum(lens)
        for (a, b), L in zip(segs, lens):
            if t <= L:
                return a[0] + (b[0] - a[0]) * t / L, a[1] + (b[1] - a[1]) * t / L
            t -= L
        return path[-1]
    for t, col in [(0.05, (255, 80, 80)), (0.12, (255, 80, 80)), (0.2, (255, 170, 60)),
                   (0.33, (255, 80, 80)), (0.41, (200, 90, 255)), (0.55, (255, 170, 60))]:
        x, y = along(t)
        d.rectangle([x - 3, y - 3, x + 3, y + 3], fill=col)
        d.point([(x - 1, y - 1), (x + 1, y - 1)], fill=(0, 0, 0))
    towers = [(60, 80, "cron"), (130, 100, "kill"), (40, 110, "grep"), (230, 100, "oom"), (230, 40, "sudo")]
    for x, y, name in towers:
        d.rectangle([x - 7, y - 7, x + 7, y + 7], outline=G, fill=(10, 40, 20))
        d.rectangle([x - 3, y - 3, x + 3, y + 3], fill=G)
        text(d, (x - 10, y + 8), name, G)
    # zap beams
    for (tx, ty), t in [((130, 100), 0.33), ((230, 100), 0.55)]:
        ex, ey = along(t)
        d.line([tx, ty, ex, ey], fill=(200, 255, 220))
    # kernel core at the exit
    d.rectangle([300, 128, 318, 152], outline=(120, 200, 255), fill=(10, 30, 60))
    text(d, (296, 116), "init", (120, 200, 255))
    # terminal HUD
    d.rectangle([0, 0, W, 16], fill=(0, 0, 0))
    text(d, (4, 2), "root@kernel:~# wave 12  load: 3.41  RAM 412M", G)
    d.rectangle([0, H - 18, W, H], fill=(0, 0, 0))
    text(d, (4, H - 16), "> fork bomb incoming... [DAEMON KEEPER concept]", (60, 160, 80))
    # scanlines
    px = img.load()
    for y in range(0, H, 2):
        for x in range(W):
            r, g, b = px[x, y]
            px[x, y] = (r * 3 // 4, g * 3 // 4, b * 3 // 4)
    save(img, "03_daemon_keeper.png")

# ---------- 4. ROOTBOUND: the garden grows back every night ----------
def rootbound():
    random.seed(4)
    img = Image.new("RGB", (W, H), (60, 120, 60))
    d = ImageDraw.Draw(img)
    T = 16
    for ty in range(0, H, T):
        for tx in range(0, W, T):
            g = random.choice([(64, 128, 62), (58, 118, 58), (70, 136, 66)])
            d.rectangle([tx, ty, tx + T - 1, ty + T - 1], fill=g)
            for _ in range(3):
                d.point((tx + random.randrange(T), ty + random.randrange(T)), fill=(90, 160, 80))
    # tilled soil plots
    for px_, py_ in [(112, 64), (128, 64), (144, 64), (112, 80), (128, 80), (144, 80)]:
        d.rectangle([px_, py_, px_ + 15, py_ + 15], fill=(110, 70, 40))
        d.line([px_ + 2, py_ + 5, px_ + 13, py_ + 5], fill=(90, 55, 30))
        d.line([px_ + 2, py_ + 10, px_ + 13, py_ + 10], fill=(90, 55, 30))
        d.rectangle([px_ + 6, py_ + 3, px_ + 9, py_ + 8], fill=(150, 220, 90))
    # creeping corrupted vines from the edges
    def vine(x, y, ang, n):
        for _ in range(n):
            nx, ny = x + 3 * math.cos(ang), y + 3 * math.sin(ang)
            d.line([x, y, nx, ny], fill=(80, 30, 90), width=3)
            d.line([x, y, nx, ny], fill=(150, 60, 160), width=1)
            if random.random() < 0.15:
                d.ellipse([nx - 2, ny - 2, nx + 2, ny + 2], fill=(220, 80, 200))
            x, y = nx, ny
            ang += random.uniform(-0.5, 0.5)
    for sx, sy, a in [(0, 30, 0.3), (0, 150, -0.4), (320, 20, 2.6), (320, 120, 3.3), (200, 180, -1.8), (60, 0, 1.3)]:
        vine(sx, sy, a, 30)
    # farmhouse
    d.rectangle([20, 70, 60, 100], fill=(170, 120, 80))
    d.polygon([(16, 70), (40, 52), (64, 70)], fill=(160, 50, 50))
    d.rectangle([36, 86, 44, 100], fill=(80, 50, 30))
    # gardener with shears
    gx, gy = 180, 96
    d.rectangle([gx, gy, gx + 6, gy + 9], fill=(60, 90, 200))
    d.rectangle([gx, gy - 5, gx + 6, gy - 1], fill=(240, 200, 160))
    d.polygon([(gx - 2, gy - 5), (gx + 3, gy - 10), (gx + 8, gy - 5)], fill=(230, 210, 120))
    d.line([gx + 7, gy + 3, gx + 13, gy - 1], fill=(200, 200, 210))
    # night tint + warm lantern pool around the gardener
    pxl = img.load()
    for y in range(H):
        for x in range(W):
            r, g, b = pxl[x, y]
            glow = max(0.0, 1 - math.hypot(x - gx - 3, y - gy) / 45)
            k = 0.45 + 0.55 * glow
            pxl[x, y] = (int(r * k + 40 * glow), int(g * k + 25 * glow), int(b * (0.75 + 0.25 * glow) + 30 * (1 - glow)))
    d = ImageDraw.Draw(img)
    # HUD
    d.rectangle([0, 0, W, 14], fill=(30, 40, 30))
    text(d, (4, 1), "NIGHT 4   sunrise in 0:42", (240, 230, 180))
    for i in range(3):
        d.rectangle([260 + i * 18, 3, 272 + i * 18, 11], fill=(150, 220, 90), outline=(40, 80, 30))
    text(d, (6, H - 14), "ROOTBOUND - concept", (120, 140, 160))
    save(img, "04_rootbound.png")

if __name__ == "__main__":
    lantern_below(); tiny_orbit(); daemon_keeper(); rootbound()
