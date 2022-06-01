def collect_exp(agent, env, max_exp_count, max_step):
    """经验收集器
    max_exp_count: 最大经验量
    max_eps：最大轮数
    max_step：最大步数
    """
    EXP, eps = [], 0
    while len(EXP) < max_exp_count:
        state = env.reset()
        for t in range(max_step):
            action = agent.forward(state, train=False)
            next_state, reward, done, _ = env.step(action)
            exp = []
            for s in state: exp.append(s)
            exp.append(action)
            EXP.append(exp)

            if done:
                eps += 1
                break
            state = next_state
            if len(EXP) >= max_exp_count:
                break
    return EXP