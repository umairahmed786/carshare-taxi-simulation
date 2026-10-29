"""Agents (passengers and taxis). Every agent's step() does, in this order:

    1. mail   = self.check_mailbox()        # FIRST: read incoming messages
    2. percepts  = self.perceive(mail)      # update beliefs from mail + environment
    3. decision  = self.deliberate(percepts)  # decide (no changes to the world)
    4. self.act(decision)                   # send messages, move, board...

Units: time in "time units", distance in "grid distance units".
"""
import math
from collections import namedtuple
from mesa import Agent
from messages import (Message, CFP, PROPOSE, REFUSE, ACCEPT, REJECT, CONFIRM, FAILURE)

EPS = 1e-9
Leg = namedtuple("Leg", "origin destination mode")       # mode: "walk" or "taxi"
Stop = namedtuple("Stop", "kind passenger cell")         # kind: "pickup" or "dropoff"


def euclid(a, b):
    """d(A, B) = sqrt((xB - xA)^2 + (yB - yA)^2)."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


class CommAgent(Agent):
    """An agent with a mailbox that can send direct messages to other agents."""

    def __init__(self, model):
        super().__init__(model)
        self.mailbox = []

    def send(self, receiver, performative, **content):
        msg = Message(performative, self, receiver, content, self.model.sim_time)
        receiver.mailbox.append(msg)
        self.model.count_message(msg)

    def check_mailbox(self):
        mail, self.mailbox = self.mailbox, []
        return mail


# --------------------------------------------------------------------------
class Passenger(CommAgent):
    """Created once at set-up with a scheduled departure time (not during the run).

    Journey time = waiting + walking + boarding + riding (contiguous, exact).
    Call states for a taxi leg: idle -> calling -> awaiting_confirm -> assigned.
    """

    def __init__(self, model, legs, departure_time):
        super().__init__(model)
        self.origin, self.destination = legs[0].origin, legs[-1].destination
        self.cell = self.origin
        self.creation_time = departure_time          # scheduled departure
        self.completion_time = None
        self.status = "scheduled"    # scheduled|planned|waiting|walking|riding|completed
        self.set_plan(legs)
        self.waiting_time = self.walking_time = self.boarding_time = self.riding_time = 0.0
        self.max_leg_wait = 0.0
        # per-leg state
        self.leg_ready_time = departure_time
        self.walk_end = None
        self.call_state, self.retry_at = "idle", 0.0
        self.round, self.proposals = 0, []
        self.n_expected = self.n_replies = 0
        self.cfp_time = None
        self._boarding_start = self._ride_start = None

    def set_plan(self, legs):
        self.legs, self.leg_index = legs, 0
        self.n_taxi_legs = sum(1 for leg in legs if leg.mode == "taxi")
        self.direct_taxi_time = sum(euclid(l.origin, l.destination) * self.model.time_per_distance
                                    for l in legs if l.mode == "taxi")

    @property
    def leg(self):
        return self.legs[self.leg_index]

    @property
    def travel_time(self):
        return self.waiting_time + self.walking_time + self.boarding_time + self.riding_time

    def is_satisfied(self):
        """Interpretable acceptability indicator, measured on the finished journey."""
        m = self.model
        return (self.max_leg_wait <= m.max_wait + EPS
                and self.riding_time <= m.max_ride_stretch * self.direct_taxi_time + EPS
                and max(self.n_taxi_legs - 1, 0) <= m.max_transfers)

    # ---- the agent cycle --------------------------------------------------
    def step(self):
        mail = self.check_mailbox()                                  # 1. mailbox first
        decision = self.deliberate(self.perceive(mail))              # 2-3
        self.act(decision)                                           # 4
        if decision["do"] == "activate":     # activation and first action in the same step
            self.act(self.deliberate(self.perceive([])))

    def perceive(self, mail):
        confirm = failure = None
        for msg in mail:
            if msg.content.get("round") != self.round:
                continue                                  # stale message from an old call
            if msg.performative == PROPOSE and self.call_state == "calling":
                self.proposals.append(msg)
                self.n_replies += 1
            elif msg.performative == REFUSE and self.call_state == "calling":
                self.n_replies += 1
            elif msg.performative == CONFIRM:
                confirm = msg
            elif msg.performative == FAILURE:
                failure = msg
        return {"now": self.model.sim_time, "confirm": confirm, "failure": failure}

    def deliberate(self, p):
        m, now = self.model, p["now"]
        if self.status == "scheduled":
            return {"do": "activate" if now >= self.creation_time - EPS else "wait"}
        if self.leg.mode == "walk":
            if self.walk_end is None:
                return {"do": "start_walk"}
            return {"do": "finish_walk" if now >= self.walk_end - EPS else "wait"}
        if self.status == "riding" or self.call_state == "assigned":
            return {"do": "wait"}
        if self.call_state == "awaiting_confirm":
            if p["confirm"]:
                return {"do": "assigned", "taxi": p["confirm"].sender}
            if p["failure"]:
                return {"do": "retry", "delay": 0.0}      # taxi no longer free: call again
            return {"do": "wait"}
        if self.call_state == "calling":
            if self.n_replies >= self.n_expected or now >= self.cfp_time + m.dt - EPS:
                return self._choose_proposal()
            return {"do": "wait"}
        return {"do": "send_cfp"} if now >= self.retry_at - EPS else {"do": "wait"}

    def _choose_proposal(self):
        """Pick the best proposal; with acceptability ON, drop unacceptable ones."""
        m = self.model
        if not self.proposals:
            return {"do": "retry", "delay": m.retry_delay}
        direct = euclid(self.leg.origin, self.leg.destination) * m.time_per_distance
        scored = []
        for msg in self.proposals:
            c = msg.content
            acceptable = (not m.acceptability) or (
                c["wait"] <= m.max_wait and c["ride"] <= m.max_ride_stretch * direct + EPS)
            if acceptable:       # passenger criterion: earliest arrival, then taxi detour
                scored.append(((c["eta"], c["added"], m.random.random()), msg))
        if not scored:
            return {"do": "retry", "delay": m.retry_delay, "unacceptable": True}
        winner = min(scored, key=lambda s: s[0])[1]
        return {"do": "accept", "winner": winner,
                "others": [x for x in self.proposals if x is not winner]}

    def act(self, d):
        m, now, what = self.model, self.model.sim_time, d["do"]
        if what == "activate":
            m.n_generated += 1
            self.status = "planned"
            m.grid.place_agent(self, self.origin)
            if m.acceptability and self.n_taxi_legs - 1 > m.max_transfers:
                m.n_transfer_refusals += 1                # too many transfers: go direct
                self.set_plan(m.plan_journey(self.origin, self.destination, multi_hop=False))
        elif what == "start_walk":
            self.walk_end = now + m.walk_time(self.leg.origin, self.leg.destination)
            self.status = "walking"
        elif what == "finish_walk":
            self.walking_time += now - self.leg_ready_time
            self.arrive(self.leg.destination, now)
        elif what == "send_cfp":
            self._send_cfp(now)
        elif what == "accept":
            w = d["winner"]
            self.send(w.sender, ACCEPT, round=self.round)
            for o in d["others"]:
                self.send(o.sender, REJECT, round=self.round)
            self.call_state = "awaiting_confirm"
        elif what == "retry":
            if d.get("unacceptable"):
                m.n_unacceptable_rounds += 1
            for o in self.proposals:
                self.send(o.sender, REJECT, round=self.round)
            self.call_state, self.retry_at = "idle", now + d["delay"]
        elif what == "assigned":
            self.call_state = "assigned"

    def _send_cfp(self, now):
        m = self.model
        targets = [t for t in m.taxis if euclid(t.fpos, self.cell) <= m.communication_range]
        self.round += 1
        self.proposals, self.n_replies, self.n_expected = [], 0, len(targets)
        self.cfp_time = now
        self.status = "waiting"
        if not targets:
            self.call_state, self.retry_at = "idle", now + m.retry_delay
            return
        self.call_state = "calling"
        m.n_cfp_rounds += 1
        for t in targets:
            self.send(t, CFP, round=self.round)

    # ---- called by the taxi (events with exact times) ---------------------
    def start_boarding(self, now):
        wait = now - self.leg_ready_time
        self.waiting_time += wait
        self.max_leg_wait = max(self.max_leg_wait, wait)
        self._boarding_start = now

    def finish_boarding(self, now):
        self.boarding_time += now - self._boarding_start
        self._ride_start = now
        self.status = "riding"
        self.model.grid.remove_agent(self)

    def alight(self, cell, now):
        self.riding_time += now - self._ride_start
        self.arrive(cell, now)

    def arrive(self, cell, now):
        m = self.model
        self.cell = cell
        self.leg_index += 1
        self.leg_ready_time, self.walk_end = now, None
        self.call_state, self.retry_at, self.proposals = "idle", now, []
        if self.leg_index == len(self.legs):
            self.status, self.completion_time = "completed", now
            if self.pos is not None:
                m.grid.remove_agent(self)
            m.record_completion(self)
            self.remove()
        else:
            self.status = "planned"
            if self.pos is None:
                m.grid.place_agent(self, cell)
            elif self.pos != cell:
                m.grid.move_agent(self, cell)


# --------------------------------------------------------------------------
class Taxi(CommAgent):
    """Answers CFPs with proposals, commits routes on ACCEPT, then drives."""

    def __init__(self, model, cell, capacity):
        super().__init__(model)
        assert 1 <= capacity <= 4
        self.capacity = capacity
        self.fpos = (float(cell[0]), float(cell[1]))
        self.onboard, self.route = [], []
        self.boarding, self.boarding_left = False, 0.0
        self.distance_traveled = self.shared_distance = 0.0
        self.operating_time = self.boarding_time_total = self.congestion_delay = 0.0

    # emissions (taxi only; passenger walking is never counted)
    @property
    def emissions_distance(self):
        return self.distance_traveled * self.model.emission_per_distance

    @property
    def emissions_boarding(self):
        return self.boarding_time_total * self.model.idle_emission_rate

    @property
    def emissions_congestion(self):
        return self.congestion_delay * self.model.idle_emission_rate

    @property
    def emissions(self):
        return self.emissions_distance + self.emissions_boarding + self.emissions_congestion

    # ---- the agent cycle --------------------------------------------------
    def step(self):
        mail = self.check_mailbox()                                  # 1. mailbox first
        self.act(self.deliberate(self.perceive(mail)))               # 2-4

    def perceive(self, mail):
        others = sum(1 for a in self.model.grid.get_cell_list_contents([self.pos])
                     if isinstance(a, Taxi) and a is not self and a.route)
        return {"others_moving": others,
                "accepts": [x for x in mail if x.performative == ACCEPT],
                "cfps": [x for x in mail if x.performative == CFP]}

    def deliberate(self, percepts):
        """Handle ACCEPTs first (they change the route), then answer CFPs."""
        m = self.model
        route, replies = list(self.route), []
        for msg in percepts["accepts"]:
            p, rnd = msg.sender, msg.content["round"]
            cand = self._best_insertion(p, route)
            if cand and (not route or cand[0] <= m.max_detour):
                route = cand[1]
                replies.append(("confirm", p, rnd, {}))
            else:
                replies.append(("failure", p, rnd, {}))
        for msg in percepts["cfps"]:
            p, rnd = msg.sender, msg.content["round"]
            cand = self._best_insertion(p, route)
            if cand and (not route or cand[0] <= m.max_detour):
                t_pick, t_board_end, t_drop = self._estimate(cand[1], p)
                info = {"wait": t_pick, "ride": t_drop - t_board_end,
                        "eta": t_drop, "added": cand[0]}
                replies.append(("propose", p, rnd, info))
            else:
                replies.append(("refuse", p, rnd, {}))
        return {"speed": 1.0 / (1.0 + m.congestion_alpha * percepts["others_moving"]),
                "route": route, "replies": replies}

    def _best_insertion(self, p, route):
        """Cheapest feasible way to add p's pickup and dropoff to `route`.
        route_change ON : stops can be inserted anywhere (even before the stop being
                          driven to) -> the taxi may divert while en route.
        route_change OFF: the next stop is committed; new stops go after it."""
        if self.boarding or (route and not self.model.route_change):
            lo = 1
        else:
            lo = 0
        pickup, dropoff = Stop("pickup", p, p.cell), Stop("dropoff", p, p.leg.destination)
        base, best = self._length(route), None
        for i in range(lo, len(route) + 1):
            for j in range(i, len(route) + 1):
                cand = route[:i] + [pickup] + route[i:j] + [dropoff] + route[j:]
                if not self._feasible(cand):
                    continue
                added = self._length(cand) - base
                if best is None or added < best[0] - EPS:
                    best = (added, cand)
        return best

    def _length(self, stops):
        total, cur = 0.0, self.fpos
        for s in stops:
            total += euclid(cur, s.cell)
            cur = s.cell
        return total

    def _feasible(self, stops):
        load = len(self.onboard)
        for s in stops:
            load += 1 if s.kind == "pickup" else -1
            if load > self.capacity:
                return False
        return True

    def _estimate(self, route, p):
        """Free-flow times from now: (pickup arrival, boarding end, dropoff arrival)."""
        m, t, cur = self.model, 0.0, self.fpos
        t_pick = t_board_end = t_drop = None
        for idx, s in enumerate(route):
            t += euclid(cur, s.cell) * m.time_per_distance
            cur = s.cell
            if s.kind == "pickup":
                if s.passenger is p:
                    t_pick = t
                t += self.boarding_left if (idx == 0 and self.boarding) else m.boarding_time
                if s.passenger is p:
                    t_board_end = t
            elif s.passenger is p:
                t_drop = t
        return t_pick, t_board_end, t_drop

    def act(self, d):
        m = self.model
        self.route = d["route"]
        for kind, p, rnd, info in d["replies"]:
            self.send(p, {"confirm": CONFIRM, "failure": FAILURE, "propose": PROPOSE,
                          "refuse": REFUSE}[kind], round=rnd, **info)
        self._drive(d["speed"])

    def _drive(self, speed):
        m = self.model
        t0, budget = m.sim_time, m.dt
        while budget > EPS and self.route:
            stop = self.route[0]
            now = t0 + m.dt - budget
            dist = euclid(self.fpos, stop.cell)
            if dist > EPS:                                    # drive
                needed = dist * m.time_per_distance / speed
                spend = min(budget, needed)
                moved = spend * speed / m.time_per_distance
                if spend >= needed - EPS:
                    self.fpos = (float(stop.cell[0]), float(stop.cell[1]))
                else:
                    f = moved / dist
                    self.fpos = (self.fpos[0] + f * (stop.cell[0] - self.fpos[0]),
                                 self.fpos[1] + f * (stop.cell[1] - self.fpos[1]))
                self.distance_traveled += moved
                if len(self.onboard) >= 2:
                    self.shared_distance += moved
                self.operating_time += spend
                self.congestion_delay += spend * (1.0 - speed)
                budget -= spend
                self._sync_grid()
            elif stop.kind == "dropoff":                      # alight (instant)
                self.onboard.remove(stop.passenger)
                self.route.pop(0)
                stop.passenger.alight(stop.cell, now)
            else:                                             # board (takes boarding_time)
                if not self.boarding:
                    self.boarding, self.boarding_left = True, m.boarding_time
                    stop.passenger.start_boarding(now)
                spend = min(budget, self.boarding_left)
                self.boarding_left -= spend
                self.boarding_time_total += spend
                self.operating_time += spend
                budget -= spend
                if self.boarding_left <= EPS:
                    self.boarding, self.boarding_left = False, 0.0
                    self.route.pop(0)
                    self.onboard.append(stop.passenger)
                    stop.passenger.finish_boarding(t0 + m.dt - budget)

    def _sync_grid(self):
        cell = (int(math.floor(self.fpos[0] + 0.5)), int(math.floor(self.fpos[1] + 0.5)))
        if cell != self.pos:
            self.model.grid.move_agent(self, cell)