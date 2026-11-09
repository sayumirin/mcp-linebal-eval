from linebal.alb import parse_alb, parse_in2, write_alb
from linebal.cases import assemplanner_valve
from linebal.tasks import generate_instance

IN2 = "7\n1\n5\n4\n3\n5\n6\n5\n1,2\n1,4\n2,3\n2,5\n4,7\n5,6\n-1,-1\n"


def test_parse_in2():
    inst = parse_in2(IN2, "Mertens")
    assert inst.n == 7
    assert inst.total_time == 29
    assert (4, 7) in inst.precedences
    assert inst.predecessors()[6] == [5]


def test_alb_roundtrip_with_constraints():
    inst = generate_instance(5, 15)
    inst.linked = [(1, 2)]
    inst.incompatible = [(3, 9)]
    inst.excluded = {4: [1, 3]}
    inst.cycle_time = 50
    back = parse_alb(write_alb(inst), inst.name)
    assert back.to_dict() == inst.to_dict()


def test_alb_accepts_space_separated_task_times_and_comments():
    text = "<number of tasks>\n3\n<cycle time> some comment\n10\n<task times>\n1 4\n2 5\n3:6\n<precedence relations>\n1,2\n2 , 3\n<end>\n"
    inst = parse_alb(text)
    assert inst.times == {1: 4, 2: 5, 3: 6}
    assert inst.precedences == [(1, 2), (2, 3)]
    assert inst.cycle_time == 10


def test_non_topological_numbering_is_accepted():
    # AssemPlanner's valve line has 12 -> 3, which the ALB spec's numbering rule would forbid
    inst = assemplanner_valve()
    assert (12, 3) in inst.precedences
    assert parse_alb(write_alb(inst)).precedences == sorted(inst.precedences)


def test_cycle_is_rejected():
    text = "<number of tasks>\n2\n<task times>\n1:1\n2:1\n<precedence relations>\n1,2\n2,1\n<end>\n"
    try:
        parse_alb(text)
    except ValueError as e:
        assert "cycle" in str(e)
    else:
        raise AssertionError("cycle not detected")
