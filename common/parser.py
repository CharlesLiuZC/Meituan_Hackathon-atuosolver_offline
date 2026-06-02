def parse_input(input_text: str) -> list:
    """Parse the input text and return a list, where each element in the list is a 4-element tuple.
    The elements in the tuple:
    1st element: score->float
    2nd element: task_id_list_str->str
    3rd element: courier->str
    4th element: willingness->float
    """
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0

    candidates = []  # (score, task_id_list_str, courier_id, willingness)
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        task_id_list_str, courier_id, score_str, willingness_str = parts[:4]
        try:
            score = float(score_str)
            willingness = float(willingness_str)
        except ValueError:
            continue
        candidates.append((score, task_id_list_str.strip(), courier_id.strip(), willingness))
    return candidates
