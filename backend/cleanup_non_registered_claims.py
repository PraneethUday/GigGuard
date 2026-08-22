#!/usr/bin/env python3
"""
One-time cleanup utility for invalid claims.

Safety defaults:
- Dry-run by default (no deletes).
- Always writes backup JSON before delete in apply mode.
- Requires BOTH --apply and --yes to perform deletions.

Eligibility rules for this cleanup:
- registered_workers.is_active = true
- registered_workers.verification_status != 'rejected'

Claim validity rules for this cleanup:
- Worker must be currently registered and eligible.
- Claim city must match the worker's registered city.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from db import supabase


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _normalize_city(city: str | None) -> str:
    value = " ".join(str(city or "").strip().lower().split())
    aliases = {
        "bangalore": "bengaluru",
        "bangaluru": "bengaluru",
        "banglore": "bengaluru",
        "bengaluru": "bengaluru",
        "new delhi": "delhi",
    }
    return aliases.get(value, value)


def fetch_registered_delivery_city_index(page_size: int = 1000) -> dict[str, str]:
    """Load eligible registered delivery IDs mapped to normalized city."""
    index: dict[str, str] = {}
    offset = 0

    while True:
        resp = (
            supabase.table("registered_workers")
            .select("delivery_id, city")
            .eq("is_active", True)
            .neq("verification_status", "rejected")
            .range(offset, offset + page_size - 1)
            .execute()
        )

        rows = resp.data or []
        if not rows:
            break

        for row in rows:
            delivery_id = str(row.get("delivery_id", "")).strip().lower()
            city = _normalize_city(row.get("city"))
            if delivery_id and city:
                index[delivery_id] = city

        if len(rows) < page_size:
            break
        offset += page_size

    return index


def fetch_claims(page_size: int = 1000) -> list[dict]:
    """Load all claims using paging."""
    claims: list[dict] = []
    offset = 0

    while True:
        resp = (
            supabase.table("claims")
            .select(
                "id,claim_number,worker_id,platform,city,trigger_id,trigger_type,"
                "payout_amount,payout_status,status,fraud_score,created_at"
            )
            .order("created_at", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
        )

        rows = resp.data or []
        if not rows:
            break

        claims.extend(rows)

        if len(rows) < page_size:
            break
        offset += page_size

    return claims


def backup_candidates(candidates: list[dict], backup_dir: Path) -> Path:
    """Write candidate claims to a timestamped backup file."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = backup_dir / f"claims_non_registered_backup_{ts}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=True, indent=2)

    return path


def delete_claims(claim_ids: list[str], batch_size: int = 200) -> int:
    """Delete claim IDs in batches; returns number of attempted deletes."""
    deleted = 0

    for batch in _chunks(claim_ids, batch_size):
        supabase.table("claims").delete().in_("id", batch).execute()
        deleted += len(batch)

    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-time safe cleanup for non-registered claims."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete rows (still requires --yes).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion action when used with --apply.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="Read page size for Supabase pagination.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Delete batch size.",
    )
    parser.add_argument(
        "--backup-dir",
        default="./backups",
        help="Directory for backup json files.",
    )
    args = parser.parse_args()

    if args.page_size < 1:
        raise ValueError("--page-size must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    print("[cleanup] Loading registered delivery IDs...")
    registered_index = fetch_registered_delivery_city_index(page_size=args.page_size)
    print(f"[cleanup] Registered eligible users: {len(registered_index)}")

    print("[cleanup] Loading claims...")
    claims = fetch_claims(page_size=args.page_size)
    print(f"[cleanup] Total claims scanned: {len(claims)}")

    candidates: list[dict] = []
    non_registered_count = 0
    city_mismatch_count = 0

    for claim in claims:
        worker_id = str(claim.get("worker_id", "")).strip().lower()
        if not worker_id:
            claim_copy = dict(claim)
            claim_copy["cleanup_reason"] = "missing_worker_id"
            candidates.append(claim_copy)
            non_registered_count += 1
            continue

        registered_city = registered_index.get(worker_id)
        if not registered_city:
            claim_copy = dict(claim)
            claim_copy["cleanup_reason"] = "non_registered_worker"
            candidates.append(claim_copy)
            non_registered_count += 1
            continue

        claim_city = _normalize_city(claim.get("city"))
        if claim_city != registered_city:
            claim_copy = dict(claim)
            claim_copy["cleanup_reason"] = "city_mismatch"
            claim_copy["registered_city"] = registered_city
            claim_copy["claim_city_normalized"] = claim_city
            candidates.append(claim_copy)
            city_mismatch_count += 1

    print(f"[cleanup] Invalid claim candidates total: {len(candidates)}")
    print(f"[cleanup]  - non-registered: {non_registered_count}")
    print(f"[cleanup]  - city-mismatch: {city_mismatch_count}")

    if not candidates:
        print("[cleanup] Nothing to clean.")
        return 0

    sample = candidates[:5]
    print("[cleanup] Sample candidates (first 5):")
    for row in sample:
        print(
            "  - "
            f"id={row.get('id')} "
            f"claim={row.get('claim_number')} "
            f"worker={row.get('worker_id')} "
            f"city={row.get('city')} "
            f"reason={row.get('cleanup_reason')}"
        )

    if not args.apply:
        print("[cleanup] Dry-run mode. No rows were deleted.")
        print("[cleanup] To apply: --apply --yes")
        return 0

    if not args.yes:
        print("[cleanup] Refusing to delete without --yes.")
        return 1

    backup_path = backup_candidates(candidates, Path(args.backup_dir))
    print(f"[cleanup] Backup written: {backup_path}")

    ids_to_delete = [str(c["id"]) for c in candidates if c.get("id")]
    deleted = delete_claims(ids_to_delete, batch_size=args.batch_size)

    print(f"[cleanup] Deleted claims: {deleted}")
    print("[cleanup] Completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
