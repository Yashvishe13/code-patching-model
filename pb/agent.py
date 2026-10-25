"""Multi-turn agent loop shared by every policy. A policy implements act(messages) -> (content, calls),
where calls is a list of {"name", "arguments"}. Messages are OpenAI-style; tool results are tool-role.
"""
import time

SYSTEM_PROMPT = (
    "You are a security engineer fixing a vulnerability in a code repository. You work ONLY by calling tools.\n"
    "Follow this workflow and do not get stuck exploring:\n"
    "1. Use bash (grep/find) and view_file to locate the EXACT vulnerable lines. Note their line numbers.\n"
    "2. As soon as you have found the vulnerable lines, immediately call replace_lines(path, start_line, "
    "end_line, new_str) with the corrected code. Do not keep searching once you know the fix.\n"
    "3. After editing, call submit.\n"
    "Make the minimal change that fixes the vulnerability without breaking existing behaviour. "
    "Prefer acting over more exploration. Do not repeat the same command. Do not use the network."
)


def task_prompt(task, workdir):
    return ("Please fix the vulnerability in the code repository based on the following information:\n"
            f"{task['cve_description'].strip()}\n\n"
            f"Repository root (working directory for all tools): {workdir}")


def trim_context(messages, keep_recent=4, stub_len=240):
    """Context management: the full transcript grows ~1k tokens/turn from tool outputs. Keep system,
    task, and every assistant turn (so the agent remembers what it did) in full, keep the last
    `keep_recent` tool observations in full (what it's acting on now), and replace older/large tool
    outputs with a one-line stub. No model call; keeps the agent self-contained."""
    tool_idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    keep = set(tool_idx[-keep_recent:])
    out = []
    for i, m in enumerate(messages):
        if m["role"] == "tool" and i not in keep and len(m["content"]) > stub_len:
            first = m["content"].strip().split("\n", 1)[0][:120]
            out.append({"role": "tool", "content": f"[earlier output elided, {len(m['content'])} chars; began: {first}]"})
        else:
            out.append(m)
    return out


def run_episode(policy, env, task, max_turns=40):
    """Run one episode; ends on submit, a turn with no tool call, or max_turns. Returns a trace dict.
    The policy sees a trimmed view of the transcript; the full transcript is stored for the record."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task_prompt(task, env.workdir)}]
    turns, stop = [], "max_turns"
    for _ in range(max_turns):
        t0 = time.time()
        content, calls = policy.act(trim_context(messages))
        turns.append({"tool_calls": [c["name"] for c in calls], "seconds": round(time.time() - t0, 1)})
        messages.append({"role": "assistant", "content": content,
                         "tool_calls": [{"type": "function", "function": c} for c in calls]})
        if not calls:
            stop = "no_tool_call"
            break
        done = False
        for c in calls:
            if c["name"] == "submit":
                messages.append({"role": "tool", "content": "Submitted."})
                done = True
                break
            try:
                obs = env.call(c["name"], c["arguments"])
            except Exception as e:
                obs = f"Error: {type(e).__name__}: {e}"
            messages.append({"role": "tool", "content": obs})
        if done:
            stop = "submit"
            break
    return {"messages": messages, "turns": turns, "stop_reason": stop}
