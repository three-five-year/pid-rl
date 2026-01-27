# adaptive_negotiation_trajectory.py
"""
Adaptive Negotiation Signals for Multi-UAV Formation Control
Based on Section III of the paper

Theory:
- Negotiation signals y_i^c evolve via Eq.(8) to achieve consensus
- Barrier functions B_ij prevent collisions
- Effective trajectory: y_i^q = y_i^c + q_i
"""

import numpy as np
from typing import List, Tuple
from util import COLLISION_THRESHOLD, DANGER_ZONE_RADIUS, SENSING_RADIUS


class AdaptiveNegotiationTrajectory:
    """
    Generate collision-free reference trajectories for multi-UAV formation

    Args:
        N: Number of agents
        adjacency_matrix: N×N communication graph (1 if connected, 0 otherwise)
        leader_access: N-dim binary vector (1 if agent can access leader)
        formation_offsets: List of N 3D offset vectors q_i
        k_gain: Control gain for consensus (default: 0.5)
        sensing_radius: Collision detection range in feet (default: 2000)
        safety_radius: Minimum safe distance in feet (default: 100)
    """

    def __init__(self,
                 N: int,
                 adjacency_matrix: np.ndarray,
                 leader_access: np.ndarray,
                 formation_offsets: List[np.ndarray],
                 k_gain: float = 0.5,
                 sensing_radius: float = SENSING_RADIUS,
                 safety_radius: float = COLLISION_THRESHOLD):

        # Validate inputs
        assert adjacency_matrix.shape == (N, N), "Adjacency matrix must be N×N"
        assert len(leader_access) == N, "Leader access must have N elements"
        assert len(formation_offsets) == N, "Must provide N formation offsets"
        assert np.sum(leader_access) >= 1, "At least one agent must access leader"

        self.N = N
        self.A = adjacency_matrix.astype(float)
        self.a0 = leader_access.astype(float)
        self.q = np.array(formation_offsets)
        self.k = k_gain
        self.R = 350  # 350ft - barrier激活范围
        self.r = safety_radius  # 100ft - 碰撞阈值
        self.N_neighbors = np.sum(self.A, axis=1)
        self.N_neighbors[self.N_neighbors == 0] = 1
        self.y_c = None
        self.y_c_dot = None

    def update_offsets(self, new_offsets: List[np.ndarray]):
        """
        Update formation offsets dynamically (e.g., during a coordinated turn).
        Args:
            new_offsets: List of (N, 3) arrays representing new q_i
        """
        assert len(new_offsets) == self.N
        self.q = np.array(new_offsets)

    def initialize(self, initial_positions: np.ndarray):
        """
        Initialize negotiation signals from agent positions

        Args:
            initial_positions: (N, 3) array of current agent positions [x, y, z]

        Formula: y_i^c(0) = p_i(0) - q_i
        This ensures y_i^q(0) = y_i^c(0) + q_i = p_i(0)
        """
        assert initial_positions.shape == (self.N, 3)
        self.y_c = initial_positions - self.q
        self.y_c_dot = np.zeros((self.N, 3))

    def barrier_function(self, d_ij: float) -> float:
        """
        碰撞避免势场函数 - 修改版

        参数:
            d_ij: 智能体间距离 (ft)

        返回:
            - d_ij < 100ft  -> 极大惩罚(碰撞)
            - 100ft ≤ d_ij < 350ft -> 平滑势场(危险区)
            - d_ij ≥ 350ft -> 安全区域,返回0
        """
        if d_ij >= self.R:  # self.R = 350ft
            return 0.0
        else:
            # 危险区域 [100, 350] ft
            # 使用平滑的Barrier函数
            numerator = (np.cos(np.pi * self.R ** 2 / d_ij ** 2) + 1) ** 2
            denominator = 2 * (d_ij ** 2 - self.r ** 2) ** 2 + 0.00001
            return numerator / denominator

    def compute_aggregate_barrier(self, agent_idx: int, y_q: np.ndarray) -> float:
        """
        Compute aggregate barrier B^a_{i,1} = Σ_{j∈N_i} B_ij

        Args:
            agent_idx: Index of current agent i
            y_q: (N, 3) array of effective trajectories for all agents

        Returns:
            Scalar barrier value (0 if safe, large if collision risk)
        """
        B_aggregate = 0.0
        y_q_i = y_q[agent_idx]

        for j in range(self.N):
            if j == agent_idx:
                continue
            if self.A[agent_idx, j] == 0:  # Not neighbors
                continue

            y_q_j = y_q[j]
            d_ij = np.linalg.norm(y_q_i - y_q_j)
            B_aggregate += self.barrier_function(d_ij)

        return B_aggregate

    def compute_dynamics(self, y_r: np.ndarray, y_r_dot: np.ndarray) -> np.ndarray:
        """
        Compute time derivatives of negotiation signals ẏ_i^c

        Eq.(8):
        ẏ_i^c = -k(1 + B^a_{i,1}) Σ_{j∈N_i} a_ij(y_i^c - y_j^c)
                + (1/N_i) Σ_{j∈N_i} a_ij(1 - a_i0) ẏ_j^c
                - k·a_i0(y_i^c - y_r) + a_i0·ẏ_r

        Args:
            y_r: (3,) leader trajectory position
            y_r_dot: (3,) leader trajectory velocity

        Returns:
            (N, 3) array of negotiation signal derivatives
        """
        if self.y_c is None:
            raise RuntimeError("Must call initialize() before computing dynamics")

        # Compute effective trajectories y_i^q = y_i^c + q_i
        y_q = self.y_c + self.q  # Shape: (N, 3)

        y_c_dot_new = np.zeros((self.N, 3))

        for i in range(self.N):
            # Term 1: Barrier-weighted consensus
            B_i = self.compute_aggregate_barrier(i, y_q)
            consensus_gain = self.k * (1 + B_i)

            consensus_term = np.zeros(3)
            for j in range(self.N):
                if self.A[i, j] > 0:
                    consensus_term += self.A[i, j] * (self.y_c[i] - self.y_c[j])

            # Term 2: Velocity consensus (critical for non-leader-accessible agents)
            velocity_consensus_term = np.zeros(3)
            if self.a0[i] == 0:  # Only for agents without leader access
                for j in range(self.N):
                    if self.A[i, j] > 0:
                        velocity_consensus_term += (self.A[i, j] / self.N_neighbors[i]) * \
                                                   self.y_c_dot[j]

            # Term 3: Leader tracking
            leader_term = np.zeros(3)
            if self.a0[i] > 0:
                leader_term = -self.k * self.a0[i] * (self.y_c[i] - y_r) + \
                              self.a0[i] * y_r_dot

            # Combine all terms
            y_c_dot_new[i] = -consensus_gain * consensus_term + \
                             velocity_consensus_term + \
                             leader_term

        self.y_c_dot = y_c_dot_new
        return y_c_dot_new

    def step(self, y_r: np.ndarray, y_r_dot: np.ndarray, dt: float):
        """
        Integrate negotiation signals forward in time

        Args:
            y_r: Leader position at current time
            y_r_dot: Leader velocity at current time
            dt: Time step in seconds
        """
        # Compute dynamics
        y_c_dot = self.compute_dynamics(y_r, y_r_dot)

        # Euler integration (can use RK4 for better accuracy)
        self.y_c += y_c_dot * dt

    def get_target_trajectories(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get effective target trajectories for all agents

        Returns:
            y_q: (N, 3) target positions
            y_q_dot: (N, 3) target velocities
        """
        y_q = self.y_c + self.q
        y_q_dot = self.y_c_dot  # Since q_i is constant
        return y_q, y_q_dot


class StraightLineLeader:
    """
    Leader trajectory: straight line motion with constant velocity

    In body frame:
    - Position: y_r(t) = y_r(0) + v_r · t · [1, 0, 0]
    - Velocity: ẏ_r(t) = v_r · [1, 0, 0]
    """

    def __init__(self,
                 initial_position: np.ndarray,
                 velocity: float = 300.0,  # ft/s
                 direction: np.ndarray = np.array([1.0, 0.0, 0.0])):
        """
        Args:
            initial_position: (3,) starting position [x, y, z] in feet
            velocity: Constant speed in ft/s
            direction: (3,) normalized direction vector
        """
        self.y0 = initial_position
        self.v = velocity
        self.direction = direction / np.linalg.norm(direction)
        self.t0 = 0.0

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get leader position and velocity at time t

        Returns:
            y_r: (3,) position
            y_r_dot: (3,) velocity
        """
        dt = t - self.t0
        y_r = self.y0 + self.v * dt * self.direction
        y_r_dot = self.v * self.direction
        return y_r, y_r_dot


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Setup 4-agent diamond formation - 扩大队形到500ft以上
    N = 4

    # Communication graph (diamond: 1-2-4, 1-3-4)
    A = np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
    ])

    # Only Agent 1 has leader access
    leader_access = np.array([1, 0, 0, 0])

    # Formation offsets (扩大到500ft以上)
    formation_offsets = [
        np.array([0.0, 0.0, 0.0]),  # Agent 1: center
        np.array([-500.0, -500.0, 0.0]),  # Agent 2: back-left
        np.array([-500.0, 500.0, 0.0]),  # Agent 3: back-right
        np.array([-1000.0, 0.0, 0.0]),  # Agent 4: rear
    ]

    # Initialize trajectory planner
    planner = AdaptiveNegotiationTrajectory(
        N=N,
        adjacency_matrix=A,
        leader_access=leader_access,
        formation_offsets=formation_offsets,
        k_gain=0.8,
        sensing_radius=SENSING_RADIUS,  # 350ft
        safety_radius=COLLISION_THRESHOLD  # 100ft
    )

    # Initialize agent positions (starting in rough formation)
    initial_positions = np.array([
        [1000.0, 0.0, -5000.0],
        [500.0, -480.0, -5000.0],
        [500.0, 480.0, -5000.0],
        [0.0, 0.0, -5000.0],
    ])
    planner.initialize(initial_positions)

    # Setup straight-line leader
    leader = StraightLineLeader(
        initial_position=np.array([1000.0, 0.0, -5000.0]),
        velocity=350.0,
        direction=np.array([1.0, 0.0, 0.0])
    )

    # Simulation parameters
    dt = 0.05
    T_sim = 20.0
    steps = int(T_sim / dt)

    # Storage for visualization
    history = {
        'time': [],
        'leader_pos': [],
        'agent_targets': [[] for _ in range(N)],
        'agent_positions': [[] for _ in range(N)],
    }

    print("=" * 70)
    print("Adaptive Negotiation Trajectory Simulation")
    print("=" * 70)
    print(f"Formation: {N} agents in diamond configuration (>500ft spacing)")
    print(f"Barrier activation range: <350ft")
    print(f"Collision threshold: <100ft")
    print(f"Simulation time: {T_sim}s, dt: {dt}s")
    print("=" * 70)

    # Main simulation loop
    for step in range(steps):
        t = step * dt

        # Get leader state
        y_r, y_r_dot = leader.get_state(t)

        # Update negotiation signals
        planner.step(y_r, y_r_dot, dt)

        # Get target trajectories
        y_q, y_q_dot = planner.get_target_trajectories()

        # Store data
        history['time'].append(t)
        history['leader_pos'].append(y_r.copy())
        for i in range(N):
            history['agent_targets'][i].append(y_q[i].copy())
            history['agent_positions'][i].append(y_q[i].copy())

        # Print progress every 2 seconds
        if step % (int(2.0 / dt)) == 0:
            print(f"\nTime: {t:.2f}s")
            print(f"Leader pos: {y_r}")
            print(f"Agent 1 target: {y_q[0]}")
            print(f"Agent 4 target: {y_q[3]}")

            # Check collision safety
            min_dist = float('inf')
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(y_q[i] - y_q[j])
                    min_dist = min(min_dist, d)

            safety_status = 'SAFE' if min_dist > 350 else ('BARRIER ACTIVE' if min_dist > 100 else 'COLLISION RISK')
            print(f"Min inter-agent distance: {min_dist:.1f} ft ({safety_status})")

    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)
