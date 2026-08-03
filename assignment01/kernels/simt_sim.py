"""问题 1.6（选做）：SIMT Simulator —— 一个 warp 的执行模拟器。

不需要 GPU

contract: 实现 run(program) -> (regs, cycles)
- warp 固定 32 个 lane，lane i 的寄存器初值为 i（int）；
- program 是指令列表，指令是元组，共三种：
    ("add", k)   active lanes 的 reg += k，1 cycle
    ("mul", k)   active lanes 的 reg *= k，1 cycle
    ("if_lt", t, then_prog, else_prog)
        reg < t 的 lane 走 then_prog，其余走 else_prog。
        模拟器先带 mask 执行 then_prog，再带 mask 的补集执行
        else_prog，然后汇合。某一支没有 active lane 时整支跳过、
        不计拍。嵌套指令照常计拍（divergence 的代价就在这里）。
        if_lt 这条指令本身不计拍，拍数只来自实际执行到的 add / mul。
- 返回值 regs 是 32 个 lane 的最终寄存器值（list），cycles 是总拍数。

通过 pytest tests/test_simt_sim.py 即为完成。
"""


def exec(program, regs, active):
    if not any(active):
        return 0

    cycles = 0

    for inst in program:
        comm = inst[0]

        if comm == "add":
            k = inst[1]
            for i, is_active in enumerate(active):
                if is_active:
                    regs[i] += k
            cycles += 1

        elif comm == "mul":
            k = inst[1]
            for i, is_active in enumerate(active):
                regs[i] *= k
            cycles += 1

        elif comm == "if_lt":
            t, then_prog, else_prog = inst[1:]

            then_mask = []
            else_mask = []

            for i, is_active in enumerate(active):
                cond = is_active and regs[i] < t

                then_mask.append(cond)
                else_mask.append(is_active and not cond)

            cycles += exec(then_prog, regs, then_mask)
            cycles += exec(else_prog, regs, else_mask)

    return cycles


def run(program):
    regs = list(range(32))
    active = [True] * 32

    cycles = exec(program, regs, active)

    return regs, cycles
