#!/usr/bin/env python3
"""
md_write_files.py

Parse a repository dump in Markdown (with sections like `### `path/to/file`` followed
by a fenced code block) and write each file to disk.

- Detects headings with backticked file paths:  ^#+ .*`path`.*
- Associates the *next* fenced code block (```...``` or ````...````) as that file's content.
- Creates parent directories as needed.
- Skips directories (headings whose backticked text ends with '/').
- Makes *.sh and *.sbatch files executable, and any file whose content starts with a shebang (#!).
- Skips overwriting existing files unless --overwrite is passed.

Examples:
  python3 md_write_files.py --root ./repo < repo_markdown.txt
  pbpaste | python3 md_write_files.py --root ./repo --overwrite
"""
import argparse
import os
import re
import stat
import sys
from typing import List, Tuple

HEADING_RE = re.compile(r'^\s{0,3}#{2,6}\s+(.*)$')          # ## ... to ###### ...
BACKTICK_PATH_RE = re.compile(r'`([^`]+?)`')                # grab `path` inside heading
FENCE_OPEN_RE = re.compile(r'^([`~]{3,})(.*)$')             # ```lang or ~~~lang or ````markdown

EXEC_SUFFIXES = ('.sh', '.sbatch')

def find_files_from_markdown(lines: List[str]) -> List[Tuple[str, str]]:
    """
    Scan lines, look for heading with backticked path; the next fenced block is the file content.
    Returns list of (path, content).
    """
    i = 0
    pairs: List[Tuple[str, str]] = []
    n = len(lines)

    while i < n:
        m_head = HEADING_RE.match(lines[i])
        if not m_head:
            i += 1
            continue

        heading_text = m_head.group(1)
        m_path = BACKTICK_PATH_RE.search(heading_text)
        if not m_path:
            i += 1
            continue

        path = m_path.group(1).strip()
        # Skip headings that denote a directory or a package section (ending with '/')
        if path.endswith('/'):
            i += 1
            continue

        # Look ahead for next fenced code block
        j = i + 1
        fence = None
        # Scan until we hit a code fence or another heading
        while j < n:
            line = lines[j]
            m_next_head = HEADING_RE.match(line)
            if m_next_head:
                # Another heading before any code fence => no content block for this path
                break

            m_fence = FENCE_OPEN_RE.match(line.strip())
            if m_fence:
                fence = m_fence.group(1)  # exact fence string (``` or ```` or ~~~)
                j += 1
                content_lines: List[str] = []
                # Collect until matching closing fence (allow leading spaces before fence)
                while j < n:
                    closer = lines[j].strip()
                    if closer.startswith(fence):
                        j += 1  # consume closing fence
                        break
                    content_lines.append(lines[j])
                    j += 1
                content = "\n".join(content_lines)
                # Ensure trailing newline (common for source files)
                if not content.endswith("\n"):
                    content += "\n"
                pairs.append((path, content))
                i = j - 1  # position just after the block; -1 because i will ++ at loop end
                break

            j += 1

        i += 1

    return pairs

def safe_join(root: str, relpath: str) -> str:
    # Normalize and prevent escaping outside root
    dest = os.path.normpath(os.path.join(root, relpath))
    root_abs = os.path.abspath(root)
    dest_abs = os.path.abspath(dest)
    if not (dest_abs == root_abs or dest_abs.startswith(root_abs + os.sep)):
        raise ValueError(f"Refusing to write outside root: {relpath}")
    return dest_abs

def ensure_mode(dest: str, content: str, set_exec: bool):
    """
    Set executable bit if:
      - file has a shebang (#!) as first two chars, or
      - file extension is in EXEC_SUFFIXES,
    and set_exec is True (default behavior).
    """
    if not set_exec:
        return
    base = os.path.basename(dest)
    is_exec_suffix = any(base.endswith(sfx) for sfx in EXEC_SUFFIXES)
    has_shebang = content.startswith("#!")
    if is_exec_suffix or has_shebang:
        try:
            st = os.stat(dest)
            os.chmod(dest, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception as e:
            print(f"[WARN] Could not set executable bit on {dest}: {e}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description="Write files from Markdown repository dump.")
    ap.add_argument(
        "--root", "-r", default=".",
        help="Output root directory (created if missing). Default: current directory."
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing files instead of skipping."
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Do not write files; just print what would be created."
    )
    ap.add_argument(
        "--no-exec", action="store_true",
        help="Do not set execute bit on scripts, even if they look executable."
    )
    args = ap.parse_args()

    text = sys.stdin.read()
    if not text.strip():
        print("[ERROR] No input read from stdin. Pipe or redirect the Markdown into this script.", file=sys.stderr)
        sys.exit(2)

    lines = text.splitlines()
    pairs = find_files_from_markdown(lines)

    if not pairs:
        print("[ERROR] No files detected in the Markdown. Make sure headings look like:  ### `path/to/file`", file=sys.stderr)
        sys.exit(1)

    # Create root if needed
    os.makedirs(args.root, exist_ok=True)

    created, skipped, overwritten = 0, 0, 0
    for relpath, content in pairs:
        try:
            dest = safe_join(args.root, relpath)
        except ValueError as e:
            print(f"[SKIP] {relpath}: {e}", file=sys.stderr)
            skipped += 1
            continue

        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)

        if os.path.exists(dest) and not args.overwrite:
            print(f"[SKIP] {relpath} (exists; use --overwrite to replace)")
            skipped += 1
            continue

        action = "OVERWRITE" if os.path.exists(dest) else "CREATE"
        if args.dry_run:
            print(f"[DRY] {action} {relpath}  ({len(content)} bytes)")
            continue

        with open(dest, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)

        ensure_mode(dest, content, set_exec=not args.no_exec)

        if action == "CREATE":
            created += 1
        else:
            overwritten += 1
        print(f"[OK] {action} {relpath}")

    if args.dry_run:
        print(f"\n[DRY SUMMARY] would create: {sum(1 for p,_ in pairs)} files")
    else:
        print(f"\n[SUMMARY] created: {created}, overwritten: {overwritten}, skipped: {skipped}")

if __name__ == "__main__":
    main()
