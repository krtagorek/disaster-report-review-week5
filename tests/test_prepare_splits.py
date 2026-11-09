"""Leakage-critical matcher checks against exhaustive enumeration."""
import itertools
from pathlib import Path
import random
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_splits import near_pairs, shingles, UnionFind


class GroupingTests(unittest.TestCase):
    def test_prefix_search_matches_exhaustive(self):
        rng = random.Random(42)
        sets = [set()]
        for _ in range(100):
            base = set(rng.sample(range(40), rng.randint(3, 30)))
            sets.extend([base, base | {rng.randrange(40)}, base - {rng.choice(sorted(base))}])
        expected = {(a, b) for a,b in itertools.combinations(range(len(sets)), 2) if sets[a] | sets[b] and len(sets[a] & sets[b])*5 >= len(sets[a] | sets[b])*4}
        self.assertEqual(set(near_pairs(sets)), expected)

    def test_numeric_updates_and_transitive_connections(self):
        self.assertEqual(shingles('The Park Fire burned 12,345 acres at 50% containment.'), shingles('The Park Fire burned 14,000 acres at 90% containment.'))
        uf = UnionFind(4)
        uf.union(0, 1)  # shared episode
        uf.union(1, 2)  # similar text in another episode
        self.assertEqual(uf.find(0), uf.find(2))
        self.assertNotEqual(uf.find(0), uf.find(3))


if __name__ == '__main__':
    unittest.main()
