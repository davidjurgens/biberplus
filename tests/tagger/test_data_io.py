"""Boundary tests for simple_split_batching and Windows-safe file I/O."""
import tempfile
import unittest
from pathlib import Path

from biberplus.tagger.data_io import (
    read_directory_of_text_files,
    read_from_file,
    simple_split_batching,
)

NON_ASCII = "café résumé naïve — 你好 — \U0001f600"


def _batches(text, size):
    return list(simple_split_batching(text, size, show_progress=False))


class TestSimpleSplitBatching(unittest.TestCase):
    def test_exact_multiple(self):
        self.assertEqual(_batches("a b c d", 2), ["a b", "c d"])

    def test_remainder(self):
        self.assertEqual(_batches("a b c", 2), ["a b", "c"])

    def test_single_token(self):
        self.assertEqual(_batches("a", 2), ["a"])

    def test_batch_size_larger_than_input(self):
        self.assertEqual(_batches("a b", 10), ["a b"])

    def test_empty_string(self):
        # split(' ') on "" yields [''], so a single empty batch is produced.
        self.assertEqual(_batches("", 2), [""])


class TestFileReads(unittest.TestCase):
    def test_read_from_file_utf8_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            fp = Path(d) / "sample.txt"
            fp.write_text(NON_ASCII, encoding="utf-8")
            self.assertEqual(read_from_file(str(fp)), NON_ASCII)

    def test_read_directory_yields_all_txt(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.txt").write_text("alpha " + NON_ASCII, encoding="utf-8")
            (Path(d) / "b.txt").write_text("beta", encoding="utf-8")
            (Path(d) / "ignore.md").write_text("not txt", encoding="utf-8")
            contents = set(read_directory_of_text_files(d))
            self.assertEqual(contents, {"alpha " + NON_ASCII, "beta"})

    def test_read_directory_without_trailing_slash(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "only.txt").write_text("solo", encoding="utf-8")
            # Path-based implementation must handle a dir path with no trailing sep.
            self.assertEqual(list(read_directory_of_text_files(d)), ["solo"])


if __name__ == "__main__":
    unittest.main()
