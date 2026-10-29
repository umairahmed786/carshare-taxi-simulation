"""Minimal direct communication: a message class and the performatives we use.

Protocol (a light Contract Net):
  passenger -> taxis      CFP      "who can take me from A to B?"
  taxi -> passenger       PROPOSE  (wait, ride, eta, added distance)  or  REFUSE
  passenger -> taxis      ACCEPT (to the chosen taxi)  /  REJECT (to the others)
  taxi -> passenger       CONFIRM (route updated)  or  FAILURE (no longer feasible)
"""
CFP, PROPOSE, REFUSE = "cfp", "propose", "refuse"
ACCEPT, REJECT, CONFIRM, FAILURE = "accept", "reject", "confirm", "failure"


class Message:
    """A direct message from one agent to another, stored in the receiver's mailbox."""

    def __init__(self, performative, sender, receiver, content=None, time=0.0):
        self.performative = performative
        self.sender = sender
        self.receiver = receiver
        self.content = content or {}
        self.time = time              # simulation time at which it was sent

    def __repr__(self):
        return (f"Message({self.performative}, {self.sender.unique_id}->"
                f"{self.receiver.unique_id}, t={self.time})")