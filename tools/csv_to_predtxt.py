"""
CSV-to-pred.txt Converter

Purpose:
- Convert a CSV file with columns [prompt, generated] into a line-oriented
  "pred.txt" compatible with eval.py. Each output line is a Python/JSON-like
  list of triples (list[list[str,str,str]]), which eval.py loads via
  ast.literal_eval.

Input Assumptions:
- The CSV has a header row with columns: "prompt", "generated".
- "prompt" follows the pattern: "Is this true: <subject> <relation> <object> ?"
  where <relation> is a single token (no spaces), <subject> can contain any
  characters, and <object> can contain spaces.
- "generated" is either "Yes" or "No" (case-insensitive). If "Yes", the triple
  is included. If "No" or parsing fails, a placeholder [["Null","Null","Null"]]
  is written to preserve line alignment with gold.txt.

Usage:
    python tools/csv_to_predtxt.py --csv path/to/input.csv --out path/to/pred.txt

Notes:
- The output preserves the exact number of input rows (excluding header),
  writing one graph per line. Each graph is currently either a single triple
  [[subject, relation, object]] if generated == Yes, or [["Null","Null","Null"]]
  otherwise, ensuring alignment with gold.txt.

Error Handling:
- Any parsing or I/O error for an individual row results in a Null placeholder
  for that row. A summary is printed at the end.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional, Tuple


PROMPT_PREFIX = "Is this true:"
NULL_TRIPLE = [["Null", "Null", "Null"]]


def parse_triple_from_prompt(prompt: str) -> Optional[Tuple[str, str, str]]:
    """Extract (subject, relation, object) from a prompt string.

    Expected format:
        "Is this true: <subject> <relation> <object> ?"

    Parsing strategy:
    - Strip leading/trailing whitespace
    - Remove the fixed English prefix and trailing question mark
    - Use a regex where relation is a single non-space token; subject is greedy but
      minimal; object is the remaining content.

    Returns None if parsing fails.
    """
    if not isinstance(prompt, str):
        return None

    text = prompt.strip()
    # Tolerate missing exact prefix by falling back to raw text
    if text.startswith(PROMPT_PREFIX):
        text = text[len(PROMPT_PREFIX):].strip()

    # Remove the final question mark if present
    if text.endswith("?"):
        text = text[:-1].strip()

    # Regex: subject (minimal greedy) + relation (no spaces) + object (rest)
    # Examples:
    #   僧 行為 見士隱抱英蓮大哭
    #   《好了歌》 作者 士隱
    pattern = re.compile(r"^(.+?)\s+(\S+)\s+(.+)$")
    m = pattern.match(text)
    if not m:
        return None
    subject, relation, obj = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    if not subject or not relation or not obj:
        return None
    return subject, relation, obj


def normalize_generated(value: str) -> Optional[bool]:
    """Map generated column text to boolean.

    Returns True for Yes, False for No, None for unknown.
    """
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v == "yes":
        return True
    if v == "no":
        return False
    return None


def convert_csv_to_pred_txt(csv_path: Path, out_path: Path) -> None:
    """Convert the input CSV into a pred.txt file expected by eval.py.

    - One output line per CSV row (excluding header).
    - If generated == Yes and prompt is parsable → write [[subject, relation, object]]
    - Else → write [["Null","Null","Null"]]
    - Uses JSON dumps to ensure proper escaping; eval.py's ast.literal_eval can read it.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written_yes = 0
    placeholders = 0
    unknown = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f_in, \
         out_path.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        if "prompt" not in reader.fieldnames or "generated" not in reader.fieldnames:
            raise ValueError("CSV must contain 'prompt' and 'generated' columns")

        for row in reader:
            total += 1
            prompt = row.get("prompt")
            gen = row.get("generated")
            gen_bool = normalize_generated(gen)

            triple: Optional[Tuple[str, str, str]] = None
            if gen_bool is True:
                triple = parse_triple_from_prompt(prompt)

            if gen_bool is True and triple is not None:
                line_obj = [[triple[0], triple[1], triple[2]]]
                written_yes += 1
            else:
                # Either generated == No, unknown, or parsing failed
                line_obj = NULL_TRIPLE
                if gen_bool is None:
                    unknown += 1
                placeholders += 1

            f_out.write(json.dumps(line_obj, ensure_ascii=False) + "\n")

    print(
        "Conversion summary: "
        f"rows={total}, yes_written={written_yes}, placeholders={placeholders}, unknown_generated={unknown}"
    )
    print(f"✓ pred.txt written to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CSV (prompt,generated) to pred.txt format")
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument("--out", required=True, help="Output pred.txt path")
    args = parser.parse_args()

    try:
        convert_csv_to_pred_txt(Path(args.csv), Path(args.out))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


