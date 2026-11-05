import re
import unittest
from pathlib import Path

RESEARCH = Path(__file__).parent.parent / "docs" / "research"


def relative_links(note):
    links = re.findall(r"\]\(([^)]+)\)", note.read_text())
    return [link.split("#")[0] for link in links if not link.startswith(("http", "#"))]


class ResearchNotesTest(unittest.TestCase):
    def test_keeps_every_relative_link_pointing_at_a_note(self):
        missing = [(note.name, link) for note in RESEARCH.rglob("*.md") for link in relative_links(note)
                   if not (note.parent / link).exists()]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
