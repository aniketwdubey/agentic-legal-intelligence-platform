"""Fetch a real public-legal corpus into data/corpus/ (opt-in, networked).

This is a stub/entry point for expanding the tiny offline fixture corpus into a
real one from PUBLIC sources only. It writes one Authority JSON per document in
the same schema the app already loads, so nothing else changes.

Public sources to wire here (all redistributable):
  * CourtListener REST API (https://www.courtlistener.com/api/) — US case law.
  * Caselaw Access Project bulk data — US case law.
  * govinfo / uscode.house.gov — US statutes (public domain).
  * CUAD (Contract Understanding Atticus Dataset) — labelled contract clauses,
    ideal for the document-intelligence + eval side.

Kept as a stub so the repo runs fully offline out of the box; implement the
source you want and run `python scripts/fetch_corpus.py --source courtlistener`.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["courtlistener", "cap", "uscode", "cuad"],
        required=True,
    )
    parser.add_argument("--out", default="data/corpus")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    raise SystemExit(
        f"[fetch_corpus] '{args.source}' ingestion is not implemented in this slice.\n"
        "Implement it to write Authority JSON (see src/legalintel/schemas.py:Authority)\n"
        f"into {args.out}/. Use only public data. The offline fixture corpus already\n"
        "lets the app, tests, and eval run without this step."
    )


if __name__ == "__main__":
    main()
