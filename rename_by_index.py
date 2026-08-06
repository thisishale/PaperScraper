import os
import re
import sys

# (directory, file extension) pairs to normalize.
TARGETS = [
    ("papers", ".pdf"),
    ("groundtruth", ".json"),
]

INDEX_PREFIX = re.compile(r"^(\d+)_")


def plan_renames(directory, ext):
    renames = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(ext):
            continue
        match = INDEX_PREFIX.match(filename)
        if not match:
            print(f"  SKIP (no leading index_): {filename}")
            continue
        index = int(match.group(1))
        new_name = f"{index:03d}{ext}"
        renames.append((filename, new_name))
    return renames


def apply_renames(directory, renames, dry_run):
    for old_name, new_name in renames:
        if old_name == new_name:
            continue
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)
        if os.path.exists(new_path) and old_name != new_name:
            print(f"  CONFLICT: {old_name} -> {new_name} (target already exists, skipped)")
            continue
        print(f"  {old_name} -> {new_name}")
        if not dry_run:
            os.rename(old_path, new_path)


def main():
    dry_run = "--apply" not in sys.argv
    print("DRY RUN (no files touched)" if dry_run else "APPLYING RENAMES")

    for directory, ext in TARGETS:
        print(f"\n{directory}/:")
        renames = plan_renames(directory, ext)
        apply_renames(directory, renames, dry_run)

    if dry_run:
        print("\nNo files were changed. Re-run with --apply to actually rename.")


if __name__ == "__main__":
    main()
