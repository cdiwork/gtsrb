"""chess-coach: turn your own games into a weakness profile.

The division of labour is deliberate: this package *measures* (Stockfish
evaluations, tactical motifs, clock usage, structural features) and writes
the numbers down. Turning those numbers into coaching prose -- what the
plans in a position were, which habit keeps costing you points -- is a job
for a language model reading the dossier this package emits.
"""

__version__ = "0.1.0"
