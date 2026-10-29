"""Taxi carshare model (Mesa >= 3.2, tested with 3.5.1). Entry point: run.py.

Switchable features for comparing configurations:
  n_passengers  : size of the demand (all passengers are created ONCE, at set-up,
                  with a scheduled departure time -> nothing is created in step()).
  shared        : False = individual taxis (capacity forced to 1),
                  True  = carshare, up to `capacity` passengers on board at once.
  capacity      : passengers allowed per taxi when shared (2..4).
  route_change  : True  = a taxi may re-route (even divert from the stop it is heading to)
                  False = the next stop is committed; new requests go after it.
  multi_hop     : True  = inter-city trips use city hubs (several taxi legs).
  boarding_time : time units per boarding (also drives boarding emissions).
  acceptability : True  = passengers reject proposals that are too slow / too detoured
                  and refuse plans with too many transfers.
"""
import heapq
import math
import random
from collections import Counter
from mesa import Model, DataCollector
from mesa.space import MultiGrid

from agents import Passenger, Taxi, Leg, euclid


class TaxiModel(Model):
    def __init__(self,
                 # --- switchable features / main scenario parameters ---
                 n_passengers=200, demand_horizon=250.0, shared=True, capacity=2,
                 route_change=True, multi_hop=True, boarding_time=0.5, acceptability=False,
                 # --- acceptability thresholds (used when acceptability=True, and for
                 #     the `satisfied_share` indicator in every run) ---
                 max_wait=8.0, max_ride_stretch=1.6, max_transfers=1,
                 # --- city / fleet ---
                 n_cities=2, city_side=4, city_gap=4, inter_city_prob=0.3, n_taxis=5,
                 # --- walking, travel time, pollution, congestion ---
                 walk_max_time=2.5, walk_time_per_cost=1.0, time_per_distance=1.0,
                 emission_per_distance=1.0, idle_emission_rate=0.5, congestion_alpha=0.3,
                 # --- negotiation ---
                 max_detour=3.0, communication_range=math.inf, retry_delay=2.0,
                 dt=1.0, taxi_positions=None, seed=None):
        super().__init__(rng=seed)
        if not shared:
            capacity = 1                          # individual taxis: one passenger at a time
        assert city_side in (2, 4) and city_gap >= 2 and 1 <= capacity <= 4
        assert time_per_distance > 0 and dt > 0 and n_taxis >= 1
        self.shared = shared and capacity > 1
        self.capacity, self.route_change, self.multi_hop = capacity, route_change, multi_hop
        self.boarding_time, self.acceptability = boarding_time, acceptability
        self.max_wait, self.max_ride_stretch, self.max_transfers = max_wait, max_ride_stretch, max_transfers
        self.walk_max_time, self.walk_time_per_cost = walk_max_time, walk_time_per_cost
        self.time_per_distance = time_per_distance
        self.emission_per_distance, self.idle_emission_rate = emission_per_distance, idle_emission_rate
        self.congestion_alpha, self.max_detour = congestion_alpha, max_detour
        self.communication_range, self.retry_delay, self.dt = communication_range, retry_delay, dt
        self.n_passengers = n_passengers
        self.sim_time = 0.0                       # own clock (Mesa has its own `model.time`)
        self._demand_rng = random.Random(seed)    # same seed -> same passengers in every config

        # ---- space: cities on a square-ish array separated by empty cells ----
        cols = math.ceil(math.sqrt(n_cities))
        rows = math.ceil(n_cities / cols)
        pitch = city_side + city_gap
        self.grid = MultiGrid(cols * pitch - city_gap, rows * pitch - city_gap, torus=False)
        self.city_of, self.city_cells = {}, []
        for c in range(n_cities):
            ox, oy = (c % cols) * pitch, (c // cols) * pitch
            cells = [(ox + i, oy + j) for i in range(city_side) for j in range(city_side)]
            self.city_cells.append(cells)
            self.city_of.update({cell: c for cell in cells})
        self.districts = [cell for cells in self.city_cells for cell in cells]
        self.hubs = [self._center_cell(cells) for cells in self.city_cells]
        self._walk_cache = {}

        # ---- counters ----
        self.n_created = self.n_generated = self.n_completed = 0
        self.n_satisfied = self.n_multi_hop = 0
        self.n_cfp_rounds = self.n_unacceptable_rounds = self.n_transfer_refusals = 0
        self.message_counts = Counter()
        self._sum_travel = self._sum_wait = self._sum_walk = self._sum_taxi_legs = 0.0

        # ---- taxis ----
        self.taxis = []
        for k in range(n_taxis):
            cell = taxi_positions[k] if taxi_positions else self.random.choice(self.districts)
            taxi = Taxi(self, cell, capacity)
            self.grid.place_agent(taxi, cell)
            self.taxis.append(taxi)

        # ---- ALL passengers are created here, once (not in step) ----
        rng = self._demand_rng
        for _ in range(n_passengers):
            origin = rng.choice(self.districts)
            c = self.city_of[origin]
            if n_cities > 1 and rng.random() < inter_city_prob:
                pool = [d for d in self.districts if self.city_of[d] != c]
            else:
                pool = [d for d in self.city_cells[c] if d != origin]
            self.create_passenger(origin, rng.choice(pool), rng.uniform(0, demand_horizon))

        self.datacollector = DataCollector(model_reporters={
            "generated": lambda m: m.n_generated,
            "completed": lambda m: m.n_completed,
            "in_system": lambda m: m.n_generated - m.n_completed,
            "avg_travel_time": lambda m: m._avg(m._sum_travel),
            "avg_waiting_time": lambda m: m._avg(m._sum_wait),
            "avg_walking_time": lambda m: m._avg(m._sum_walk),
            "avg_taxi_legs": lambda m: m._avg(m._sum_taxi_legs),
            "multi_hop_completed": lambda m: m.n_multi_hop,
            "satisfied_share": lambda m: m._avg(m.n_satisfied),
            "taxi_occupancy": lambda m: sum(len(t.onboard) for t in m.taxis) / len(m.taxis),
            "total_taxi_distance": lambda m: sum(t.distance_traveled for t in m.taxis),
            "shared_distance": lambda m: sum(t.shared_distance for t in m.taxis),
            "total_taxi_operating_time": lambda m: sum(t.operating_time for t in m.taxis),
            "congestion_delay": lambda m: sum(t.congestion_delay for t in m.taxis),
            "emissions_distance": lambda m: sum(t.emissions_distance for t in m.taxis),
            "emissions_boarding": lambda m: sum(t.emissions_boarding for t in m.taxis),
            "emissions_congestion": lambda m: sum(t.emissions_congestion for t in m.taxis),
            "total_taxi_emissions": lambda m: sum(t.emissions for t in m.taxis),
            "n_messages": lambda m: sum(m.message_counts.values()),
            "n_cfp_rounds": lambda m: m.n_cfp_rounds,
            "unacceptable_rounds": lambda m: m.n_unacceptable_rounds,
            "transfer_refusals": lambda m: m.n_transfer_refusals,
        })
        self.datacollector.collect(self)

    # ---- geometry ---------------------------------------------------------
    @staticmethod
    def _center_cell(cells):
        cx = sum(c[0] for c in cells) / len(cells)
        cy = sum(c[1] for c in cells) / len(cells)
        return min(cells, key=lambda c: (euclid(c, (cx, cy)), c))

    @staticmethod
    def step_walking_cost(a, b):
        """Adjacent cells: orthogonal = 2 x Euclidean, diagonal = 0.5 x Euclidean."""
        orthogonal = (a[0] == b[0]) or (a[1] == b[1])
        return euclid(a, b) * (2.0 if orthogonal else 0.5)

    def walking_cost(self, a, b):
        """Cheapest walking cost inside one city (Dijkstra); infinite between cities."""
        if a == b:
            return 0.0
        if self.city_of[a] != self.city_of[b]:
            return math.inf
        if (a, b) in self._walk_cache:
            return self._walk_cache[(a, b)]
        best, heap = {a: 0.0}, [(0.0, a)]
        while heap:
            cost, cur = heapq.heappop(heap)
            if cur == b:
                break
            if cost > best[cur]:
                continue
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nxt = (cur[0] + dx, cur[1] + dy)
                    if (dx or dy) and self.city_of.get(nxt) == self.city_of[a]:
                        c = cost + self.step_walking_cost(cur, nxt)
                        if c < best.get(nxt, math.inf):
                            best[nxt] = c
                            heapq.heappush(heap, (c, nxt))
        self._walk_cache[(a, b)] = best[b]
        return best[b]

    def walk_time(self, a, b):
        """Walking COST is interpreted as simulated time (x walk_time_per_cost)."""
        return self.walking_cost(a, b) * self.walk_time_per_cost

    def plan_journey(self, origin, destination, multi_hop=None):
        """Legs of a journey. With multi_hop, inter-city trips go through the city hubs."""
        multi_hop = self.multi_hop if multi_hop is None else multi_hop

        def local(a, b):
            if a == b:
                return []
            return [Leg(a, b, "walk" if self.walk_time(a, b) <= self.walk_max_time else "taxi")]

        if self.city_of[origin] == self.city_of[destination]:
            return local(origin, destination)
        if not multi_hop:
            return [Leg(origin, destination, "taxi")]
        h_o, h_d = self.hubs[self.city_of[origin]], self.hubs[self.city_of[destination]]
        return local(origin, h_o) + [Leg(h_o, h_d, "taxi")] + local(h_d, destination)

    # ---- passengers & bookkeeping ------------------------------------------
    def create_passenger(self, origin, destination, departure_time=None):
        """Used at set-up (and by tests). The passenger stays 'scheduled' until its time."""
        assert origin != destination
        t = self.sim_time if departure_time is None else departure_time
        self.n_created += 1
        return Passenger(self, self.plan_journey(origin, destination), t)

    def count_message(self, msg):
        self.message_counts[msg.performative] += 1

    def record_completion(self, p):
        assert abs(p.travel_time - (p.completion_time - p.creation_time)) < 1e-6
        self.n_completed += 1
        self._sum_travel += p.travel_time
        self._sum_wait += p.waiting_time
        self._sum_walk += p.walking_time
        self._sum_taxi_legs += p.n_taxi_legs
        self.n_multi_hop += p.n_taxi_legs >= 2
        self.n_satisfied += p.is_satisfied()

    def _avg(self, total):
        """Average over COMPLETED passengers only (NaN if none yet)."""
        return total / self.n_completed if self.n_completed else float("nan")

    @property
    def all_done(self):
        return self.n_completed == self.n_created

    # ---- main loop ----------------------------------------------------------
    def step(self):
        # No passenger is created here: scheduled passengers activate themselves.
        self.agents.select(agent_type=Passenger).shuffle_do("step")
        self.agents.select(agent_type=Taxi).shuffle_do("step")
        self.sim_time += self.dt
        self.datacollector.collect(self)

    def run(self, max_steps=1500):
        """Run until every passenger has completed (or max_steps)."""
        for _ in range(max_steps):
            if self.all_done:
                break
            self.step()
        return self