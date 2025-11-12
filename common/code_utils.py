def remove_comments_keep_lines(code: str) -> str:
    """
    Removes Python comments while keeping line structure to avoid indentation issues.
    """
    lines = code.splitlines(True)
    out = []
    for line in lines:
        if "#" in line:
            idx = line.index("#")
            before = line[:idx].rstrip()
            if before.strip() == "":
                out.append("\n")
            else:
                out.append(before + " \n")
        else:
            out.append(line)
    return "".join(out)


def code_runs_ok(code: str) -> bool:
    try:
        compiled = compile(code, "<string>", "exec")
        exec(compiled, {})
        return True
    except Exception as e:
        print("Error:", e)
        return False
