import json
import re
from tqdm import tqdm

INPUT_FILE = "./mental_health_vi.json"
OUTPUT_FILE = "./mental_health_vi_final.json"

# all_assistant_turns: sinh 1 sample cho mỗi lượt trả lời của Trợ lý trong hội thoại dài.
# last_assistant_turn: chỉ giữ sample ở lượt trả lời cuối cùng của Trợ lý.
DIALOGUE_STRATEGY = "all_assistant_turns"


ROLE_USER = "Người dùng"
ROLE_ASSISTANT = "Trợ lý"
ROLE_PATTERN = re.compile(r"(Người dùng:|Trợ lý:)")


def normalize_text(text):
    return " ".join((text or "").strip().split())


def parse_dialogue_turns(text):
    """Parse chuỗi hội thoại thành list[(role, content)]."""
    if not text:
        return []

    parts = ROLE_PATTERN.split(text)
    turns = []

    for i in range(1, len(parts), 2):
        role_token = parts[i].strip()
        content = (parts[i + 1] if i + 1 < len(parts) else "").strip()
        if not content:
            continue
        role = ROLE_USER if role_token.startswith(ROLE_USER) else ROLE_ASSISTANT
        turns.append((role, normalize_text(content)))

    return turns


def format_history(turns):
    return "\n".join(f"{role}: {content}" for role, content in turns)


def build_samples_from_dialogue(turns, strategy=DIALOGUE_STRATEGY):
    """Sinh sample SFT từ hội thoại nhiều lượt."""
    samples = []
    history = []

    for role, content in turns:
        if role == ROLE_ASSISTANT:
            instruction = format_history(history).strip()
            output = content.strip()
            if instruction and output:
                samples.append(
                    {
                        "instruction": instruction,
                        "output": output,
                    }
                )
        history.append((role, content))

    if strategy == "last_assistant_turn" and samples:
        return [samples[-1]]

    return samples


def build_sample_from_short_pair(example):
    instruction = normalize_text(example.get("instruction", ""))
    output = normalize_text(example.get("output", ""))
    if instruction and output:
        return [{"instruction": instruction, "output": output}]
    return []


def process_example(example):
    instruction_text = (example.get("instruction") or "").strip()

    # Nếu instruction có marker hội thoại, ưu tiên parse thành multi-turn.
    if ROLE_PATTERN.search(instruction_text):
        turns = parse_dialogue_turns(instruction_text)
        samples = build_samples_from_dialogue(turns)
        if samples:
            return samples, "dialogue"

    # Nếu là cặp ngắn chuẩn instruction/output.
    short_samples = build_sample_from_short_pair(example)
    if short_samples:
        return short_samples, "short_pair"

    return [], "skipped"


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_samples = []
    seen = set()
    stats = {
        "short_pair": 0,
        "dialogue": 0,
        "skipped": 0,
        "duplicates_removed": 0,
    }

    for ex in tqdm(data, desc="Đang xử lý toàn bộ dataset"):
        samples, source_type = process_example(ex)
        stats[source_type] += 1

        for s in samples:
            key = s["instruction"] + " || " + s["output"]
            if key in seen:
                stats["duplicates_removed"] += 1
                continue
            seen.add(key)
            all_samples.append(s)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"Tổng sample cuối: {len(all_samples):,}")
    print(
        "Nguồn dữ liệu - "
        f"short_pair: {stats['short_pair']:,}, "
        f"dialogue: {stats['dialogue']:,}, "
        f"skipped: {stats['skipped']:,}"
    )
    print(f"Đã loại trùng: {stats['duplicates_removed']:,}")
    print(f"Đã lưu tại: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
