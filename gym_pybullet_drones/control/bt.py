"""Minimal behavior tree (Selector/Sequence/Condition/Action).

Zero dependencies, tick-returns SUCCESS/FAILURE, and every tick records
the branch path so you can see exactly which behavior fired per drone.
"""
from gym_pybullet_drones.utils.tlog import get_logger

SUCCESS, FAILURE = True, False


class Node:
    def tick(self, bb): raise NotImplementedError


class Selector(Node):
    """Priority-ordered: first child to succeed wins (OR with short-circuit)."""
    def __init__(self, name, children):
        self.name, self.children = name, children

    def tick(self, bb):
        for c in self.children:
            if c.tick(bb) == SUCCESS:
                bb['path'].append(f"[{self.name}>{getattr(c, 'name', c.__class__.__name__)}]")
                return SUCCESS
        return FAILURE


class Sequence(Node):
    """All children must succeed in order (AND)."""
    def __init__(self, name, children):
        self.name, self.children = name, children

    def tick(self, bb):
        for c in self.children:
            if c.tick(bb) != SUCCESS:
                return FAILURE
        bb['path'].append(f"<{self.name}>")
        return SUCCESS


class Condition(Node):
    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def tick(self, bb):
        return SUCCESS if self.fn(bb) else FAILURE


class Action(Node):
    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def tick(self, bb):
        return self.fn(bb)


def build_defense_behavior_tree(agent):
    """Priority: WARMUP > SAFETY > ENGAGE > PATROL."""
    log = get_logger("bt")

    def act_warmup(bb):
        bb['rpm'] = agent._act_warmup(bb)
        return SUCCESS

    def act_escape(bb):
        bb['rpm'] = agent._act_escape(bb)
        return SUCCESS

    def act_intercept(bb):
        bb['rpm'] = agent._act_intercept(bb)
        return SUCCESS

    def act_patrol(bb):
        bb['rpm'] = agent._act_patrol(bb)
        return SUCCESS

    root = Selector("defense", [
        Sequence("warmup", [Condition("t<1s", lambda bb: bb['time_now'] < 1.0),
                            Action("hover@asset", act_warmup)]),
        Sequence("safety", [Condition("ally-too-close", lambda bb: agent._cond_safety(bb)),
                            Action("escape", act_escape)]),
        Sequence("engage", [Condition("target-assigned", lambda bb: bb['target'] is not None),
                            Action("pn-intercept", act_intercept)]),
        Action("patrol-orbit", act_patrol),
    ])

    def tick_wrapper(bb):
        bb.setdefault('path', [])
        status = root.tick(bb)
        path = "".join(bb['path'])
        if path != getattr(agent, '_last_bt_path', None):
            log.log("bt_branch", drone=int(agent.id), path=path or "patrol-orbit")
            agent._last_bt_path = path
        bb['path'] = []
        return status

    return tick_wrapper
