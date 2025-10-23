import math
import logging
from typing import Union, Iterable, Sequence, List


# ======================================================
# =============== 幾何核心：2D 基礎 =====================
# ======================================================
class Base2D:
    def __init__(self, x: Union[float, Sequence[float]], y: float = None):
        if y is None:
            if isinstance(x, (list, tuple)) and len(x) == 2:
                self.x, self.y = x
            else:
                raise ValueError("輸入必須是 (x, y) 或 [x, y]")
        else:
            self.x, self.y = x, y

    def __iter__(self):
        yield self.x
        yield self.y

    def __repr__(self):
        return f"{self.__class__.__name__}({self.x}, {self.y})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return math.isclose(
            self.x, other.x, rel_tol=1e-9, abs_tol=1e-12
        ) and math.isclose(self.y, other.y, rel_tol=1e-9, abs_tol=1e-12)

    def __hash__(self):
        return hash((round(self.x, 12), round(self.y, 12)))


# ======================================================
# ==================== 點與向量 =========================
# ======================================================
# 點物件
class Point(Base2D):

    ORIGIN: "Point" = None

    @staticmethod
    def to_point(p: Union["Point", Sequence[float]]) -> "Point":
        if isinstance(p, Point):
            return p
        if isinstance(p, (list, tuple)) and len(p) == 2:
            return Point(*p)
        raise ValueError("輸入必須是 Point 或 [x, y]")

    def __add__(self, other: Union["Vector", Sequence[float]]) -> "Point":
        o = Vector.to_vector(other)
        return Point(self.x + o.x, self.y + o.y)

    def __sub__(self, other: Union["Point", Sequence[float]]) -> "Vector":
        o = Point.to_point(other)
        return Vector(self.x - o.x, self.y - o.y)

    def distance_to(self, other: Union["Point", Sequence[float]]) -> float:
        o = Point.to_point(other)
        return math.hypot(self.x - o.x, self.y - o.y)


# 常見共用屬性，使用同一物件。
Point.ORIGIN = Point(0, 0)


# 向量物件
class Vector(Base2D):
    @staticmethod
    def to_vector(v: Union["Vector", "Point", Sequence[float]]) -> "Vector":
        if isinstance(v, Vector):
            return v
        if isinstance(v, Point):
            return Vector(v.x, v.y)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return Vector(*v)
        raise ValueError("輸入必須是 Vector 或 [x, y]")

    # 運算
    def __add__(self, other) -> "Vector":
        o = Vector.to_vector(other)
        return Vector(self.x + o.x, self.y + o.y)

    def __sub__(self, other) -> "Vector":
        o = Vector.to_vector(other)
        return Vector(self.x - o.x, self.y - o.y)

    def __mul__(self, scalar: float) -> "Vector":
        if isinstance(scalar, (int, float)):
            return Vector(self.x * scalar, self.y * scalar)
        raise TypeError("只能與數字相乘")

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> "Vector":
        if not isinstance(scalar, (int, float)):
            raise TypeError("只能與數字相除")
        if scalar == 0:
            raise ZeroDivisionError("除以零錯誤")
        return Vector(self.x / scalar, self.y / scalar)

    def __neg__(self) -> "Vector":
        return Vector(-self.x, -self.y)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def dot(self, other) -> float:
        o = Vector.to_vector(other)
        return self.x * o.x + self.y * o.y

    def cross(self, other) -> float:
        o = Vector.to_vector(other)
        return self.x * o.y - self.y * o.x


# ======================================================
# ==================== 幾何圖形 =========================
# ======================================================
# 線物件
class Line:
    def __init__(
        self, p1: Union[Point, Sequence[float]], p2: Union[Point, Sequence[float]]
    ):
        # 支援Point物件以及Sequence[float]，並自動轉為Point物件。
        self.point1 = Point.to_point(p1)
        self.point2 = Point.to_point(p2)
        self.vector: Vector = self.point2 - self.point1

    # 是否與傳入之Line平行
    def is_parallel(self, other: "Line", tol: float = 1e-10) -> bool:
        return math.isclose(self.vector.cross(other.vector), 0.0, abs_tol=tol)

    # 是否與傳入之Line垂直
    def is_perpendicular(self, other: "Line", tol: float = 1e-10) -> bool:
        return math.isclose(self.vector.dot(other.vector), 0.0, abs_tol=tol)


# 圓物件
class Circle:
    def __init__(self, o: Union[Point, Sequence[float]], r: float):
        if r <= 0:
            raise ValueError("半徑必須大於 0")
        self.o = Point.to_point(o)
        self.r = r

    # 每次呼叫area會根據當前半徑計算面積
    @property
    def area(self) -> float:
        return math.pi * self.r**2

    # 與傳入支圓物件是否相交
    def intersects(self, other: "Circle", tol: float = 1e-12) -> bool:
        return self.o.distance_to(other.o) <= (self.r + other.r + tol)

    def point_position(
        self, p: Union[Point, Sequence[float]], tol: float = 1e-12
    ) -> str:
        pt = Point.to_point(p)
        d = self.o.distance_to(pt)
        if math.isclose(d, self.r, abs_tol=tol):
            return "on"
        elif d < self.r:
            return "inside"
        else:
            return "outside"


# 凸多邊形物件
class Polygon:
    def __init__(self, *points: Union[Point, Iterable[float]]):
        # 支援單個Iterable[Point]、Iterable[Iterable[float]]或是複數個Iterable[float] 或是 複數個points
        input_points: List[Union[Point, Sequence[float]]]
        if len(points) == 1 and isinstance(points[0], (list, tuple, set)):
            input_points = list(points[0])
        else:
            input_points = list(points)

        n = len(input_points)
        if n < 3:
            raise ValueError("多邊形至少需要 3 個點")

        self.points: List[Point] = [Point.to_point(p) for p in input_points]

        # 偵測是否有重複的點
        seen = set()
        for p in self.points:
            if p in seen:
                raise ValueError(f"偵測到重複的點：{p}")
            seen.add(p)

        total_vector = sum((Vector(p.x, p.y) for p in self.points), Vector(0, 0))
        centroid_vector = total_vector / n
        self.centroid: Point = Point(centroid_vector.x, centroid_vector.y)
        self.points.sort(
            key=lambda p: math.atan2(p.y - self.centroid.y, p.x - self.centroid.x)
        )
        self.perimeter: float = sum(
            self.points[i].distance_to(self.points[(i + 1) % n]) for i in range(n)
        )


# Task2
# ======================================================
# ==================== 遊戲邏輯 =========================
# ======================================================
class Enemy:

    _counter = 0

    def __init__(
        self,
        point: Union[Point, Sequence[float]],
        hp: int,
        vector: Union[Point, Vector, Sequence[float]],
    ):
        Enemy._counter += 1
        self.id = f"E{Enemy._counter}"
        self.position = Point.to_point(point)
        self.hp = hp
        self.vector = Vector.to_vector(vector)
        self.is_alive = True
        self.dead_at = None

    def move(self):
        if self.is_alive:
            self.position += self.vector
            logger.debug(
                f"{self.id} 移動到 ({self.position.x:.2f}, {self.position.y:.2f})"
            )

    def status_update(self):
        self.hp = max(self.hp, 0)
        if self.hp == 0:
            self.is_alive = False
            self.dead_at = self.position
            logger.debug(
                f"{self.id} 死亡於 ({self.dead_at.x:.2f}, {self.dead_at.y:.2f})"
            )


class Tower:

    _counter_basic = 0
    _counter_advanced = 0

    DEFAULT_RADIUS = None
    DEFAULT_ATTACK = None

    def __init__(
        self, center: Union[Point, Sequence[float]], r: float = None, attack: int = None
    ):
        self.center = Point.to_point(center)
        self.r = r if r is not None else self.DEFAULT_RADIUS
        self.attack = attack if attack is not None else self.DEFAULT_ATTACK
        if self.r is None or self.attack is None:
            raise ValueError("必須指定半徑與攻擊力")
        self.range_circle = Circle(self.center, self.r)

        if isinstance(self, BasicTower):
            Tower._counter_basic += 1
            self.id = f"T{Tower._counter_basic}"
        elif isinstance(self, AdvancedTower):
            Tower._counter_advanced += 1
            self.id = f"A{Tower._counter_advanced}"
        else:
            self.id = "Unknown"

    def attack_enemy(self, enemy: "Enemy"):
        if self.range_circle.point_position(enemy.position) != "outside":
            if enemy.is_alive:
                enemy.hp -= self.attack
                logger.debug(
                    f"{self.id} 對 {enemy.id} 攻擊 {self.attack} 點傷害，HP={enemy.hp}"
                )


class BasicTower(Tower):
    DEFAULT_RADIUS = 2
    DEFAULT_ATTACK = 2


class AdvancedTower(Tower):
    DEFAULT_RADIUS = 4
    DEFAULT_ATTACK = 4


# ======================================================
# ================= 遊戲流程控制 ========================
# ======================================================
def next_round(game, round_number) -> bool:

    logger.debug(f"\n=== Round {round_number + 1} ===")

    def enemy_move(enemies: list):
        for enemy in enemies:
            enemy.move()

    def tower_attack(enemies: list, towers: list):
        for tower in towers:
            for enemy in enemies:
                tower.attack_enemy(enemy)

    def status_update(enemies: list) -> bool:
        is_continue = True
        enemies_hp = 0
        for enemy in enemies:
            enemy.status_update()
            enemies_hp += enemy.hp
        is_continue = False if enemies_hp == 0 else True
        return is_continue

    enemy_move(game["enemies"])
    tower_attack(game["enemies"], game["towers"])
    return status_update(game["enemies"])


def create_default_game():
    return {
        "total_round": 10,
        "enemies": [
            Enemy((-10, 2), 10, (2, -1)),
            Enemy((-8, 0), 10, (3, 1)),
            Enemy((-9, -1), 10, (3, 0)),
        ],
        "towers": [
            BasicTower((-3, 2)),
            BasicTower((-1, -2)),
            BasicTower((4, 2)),
            BasicTower((7, 0)),
            AdvancedTower((1, 1)),
            AdvancedTower((4, -3)),
        ],
    }


def gameStart(game_setting=None):
    if game_setting is None:
        game_setting = create_default_game()
    print("歡迎進入遊戲，遊戲以展示模式開始。")

    for i in range(game_setting["total_round"]):
        if not next_round(game_setting, i):
            logger.info(f"所有敵人已在回合 {i + 1} 被消滅。")
            break

    print("\n--- 遊戲結束 ---")
    print("敵人最終狀態：")
    for enemy in game_setting["enemies"]:
        print(
            f"{enemy.id}: Position=({enemy.position.x:.2f}, {enemy.position.y:.2f}), "
            f"HP={enemy.hp}"
        )


# ======================================================
# ===================== Task1 ==========================
# ======================================================
print("\n-----以Point物件作為參數----")
p1 = Point(-6, 1)
p2 = Point(2, 4)
p3 = Point(-6, -1)
p4 = Point(2, 2)
p5 = Point(-4, -4)
p6 = Point(-1, 6)
p7 = Point(6, 3)
p8 = Point(8, 1)
p9 = Point(2, 0)
p10 = Point(5, -1)
p11 = Point(-1, -2)
p12 = Point(4, -4)

Line_A = Line(p1, p2)
Line_B = Line(p3, p4)
Line_C = Line(p5, p6)
Circle_A = Circle(p7, 2)
Circle_B = Circle(p8, 1)
Polygon_A = Polygon(p9, p10, p11, p12)

print(
    f"Are Line A and Line B parallel? {Line_A.is_parallel(Line_B)}",
    f"Are Line C and Line A perpendicular? {Line_C.is_perpendicular(Line_A)}",
    f"Print the area of Circle A. {Circle_A.area}",
    f"Do Circle A and Circle B intersect? {Circle_A.intersects(Circle_B)}",
    f"Print the perimeter of Polygon A. {Polygon_A.perimeter}",
    sep="\n",
)

print("\n----以(x,y)作為參數----")
Line_A = Line((-6, 1), (2, 4))
Line_B = Line((-6, -1), (2, 2))
Line_C = Line((-4, -4), (-1, 6))
Circle_A = Circle((6, 3), 2)
Circle_B = Circle((8, 1), 1)
Polygon_A = Polygon((2, 0), (4, -4), (-1, -2), (5, -1))

print(
    f"Are Line A and Line B parallel? {Line_A.is_parallel(Line_B)}",
    f"Are Line C and Line A perpendicular? {Line_C.is_perpendicular(Line_A)}",
    f"Print the area of Circle A. {Circle_A.area}",
    f"Do Circle A and Circle B intersect? {Circle_A.intersects(Circle_B)}",
    f"Print the perimeter of Polygon A. {Polygon_A.perimeter}\n",
    sep="\n",
)


# ======================================================
# ===================== Task2 ==========================
# ======================================================
# === 設定 logging ===
logger = logging.getLogger("TowerDefense")
logger.setLevel(logging.INFO)  # 預設層級，可改成 DEBUG 看詳細過程

# 建立輸出格式與 handler
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
gameStart()
