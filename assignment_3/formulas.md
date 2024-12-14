# Mathematical Formulas for Random Point and Orientation

## Important Positions
- GK: (-14, 0, 0)
- Left Post (from GKs View): (-15, 1, 0)
- Right Post (from GKs View): (-15, -1, 0)
	- Goal Line: (-15, [-1, 1], 0)
- Furthest Spawnpoint on Left Lane (from GKs View): (-6, 10, 0)
- Furthest Spawnpoint on Right Lane (from GKs View): (-6, -10, 0)
- Closest Spawnpoint on Left Lane (from GKs View): (-10, 6, 0)
- Closest Spawnpoint on Right Lane (from GKs View): (-10, -6, 0)
- Intersection of furthest points towards respective posts: (-16, 0, 0)

## Random Point Within Area

To calculate a random point within the defined area:

### 1. Radius Calculation

$$
R_{furthest} = \sqrt{(x_{furthest} - x_{center})^2 + (y_{furthest} - y_{center})^2}
$$
$$
R_{closest} = \sqrt{(x_{closest} - x_{center})^2 + (y_{closest} - y_{center})^2}
$$

### 2. Angle Calculation

$$
\theta_{left} = \arctan2(y_{furthest,left} - y_{center}, x_{furthest,left} - x_{center})
$$
$$
\theta_{right} = \arctan2(y_{furthest,right} - y_{center}, x_{furthest,right} - x_{center})
$$
If \(\theta_{left} < \theta_{right}\), swap \(\theta_{left}\) and \(\theta_{right}\).

### 3. Generate Random Radius and Angle

$$
R_{random} \in [R_{closest}, R_{furthest}]
$$
$$
\theta_{random} \in [\theta_{right}, \theta_{left}]
$$

### 4. Cartesian Coordinates

$$
x = x_{center} + R_{random} \cdot \cos(\theta_{random})
$$
$$
y = y_{center} + R_{random} \cdot \sin(\theta_{random})
$$

---

## Orientation Towards Goal Line

To calculate the orientation towards a random point on the goal line:

### 1. Random Point on Goal Line

$$
y_{goal} \in [y_{goal,min}, y_{goal,max}], \quad x_{goal} = x_{goal,line}
$$

### 2. Angle to Goal

$$
\theta_{goal} = \arctan2(y_{goal} - y_{spawn}, x_{goal} - x_{spawn})
$$
Normalize \(\theta_{goal}\) to be in the range \([0, 360)\):

$$
\theta_{goal} = \begin{cases} 
\theta_{goal} + 360 & \text{if } \theta_{goal} < 0 \\
\theta_{goal} & \text{otherwise} 
\end{cases}
$$

---