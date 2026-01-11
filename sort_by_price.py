#!/usr/bin/env python3
"""
MTG Card Price Sorter

Reads a list of MTG card names from a text file and sorts them by price
using either TCGPlayer (via Scryfall) or Card Kingdom APIs.

Usage:
    python sort_by_price.py cards.txt
    python sort_by_price.py cards.txt --source cardkingdom
    python sort_by_price.py cards.txt --descending
    python sort_by_price.py cards.txt --output sorted_cards.txt
    python sort_by_price.py cards.txt --threshold 1.00  # Split at $1.00
"""

import argparse
import json
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import requests


def normalize_name(name: str) -> str:
    """Normalize card name for lookup (strip accents, lowercase)."""
    # NFD decomposes characters, then filter out combining marks (accents)
    normalized = unicodedata.normalize("NFD", name)
    ascii_name = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    return ascii_name.strip().lower()


# API endpoints
SCRYFALL_API = "https://api.scryfall.com/cards/named"
CARDKINGDOM_API = "https://api.cardkingdom.com/api/pricelist"

# Rate limiting
REQUEST_DELAY = 0.1  # 100ms between requests (Scryfall asks for 50-100ms)

# Cache settings
CACHE_FILE = Path(__file__).parent / ".price_cache.json"
CK_PRICELIST_FILE = Path(__file__).parent / ".ck_pricelist.json"
CACHE_MAX_AGE = timedelta(hours=24)


@dataclass
class CardPrice:
    name: str
    price_usd: float | None
    price_display: str
    error: str | None = None


@dataclass
class CacheEntry:
    name: str
    price_usd: float | None
    fetched_at: str  # ISO format timestamp


def load_cache() -> dict[str, CacheEntry]:
    """Load the price cache from disk."""
    if not CACHE_FILE.exists():
        return {}

    with open(CACHE_FILE, "r") as f:
        data = json.load(f)

    cache = {}
    for key, entry in data.items():
        cache[key] = CacheEntry(
            name=entry["name"],
            price_usd=entry["price_usd"],
            fetched_at=entry["fetched_at"],
        )
    return cache


def save_cache(cache: dict[str, CacheEntry]) -> None:
    """Save the price cache to disk."""
    data = {}
    for key, entry in cache.items():
        data[key] = {
            "name": entry.name,
            "price_usd": entry.price_usd,
            "fetched_at": entry.fetched_at,
        }

    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def is_cache_valid(entry: CacheEntry) -> bool:
    """Check if a cache entry is still valid (less than 24 hours old)."""
    fetched_at = datetime.fromisoformat(entry.fetched_at)
    return datetime.now() - fetched_at < CACHE_MAX_AGE


def get_cached_price(card_name: str, cache: dict[str, CacheEntry], source: str = "tcgplayer") -> CardPrice | None:
    """Get a card's price from cache if available and valid."""
    key = f"{source}:{card_name.strip().lower()}"
    if key not in cache:
        return None

    entry = cache[key]
    if not is_cache_valid(entry):
        return None

    if entry.price_usd is not None:
        return CardPrice(
            name=entry.name,
            price_usd=entry.price_usd,
            price_display=f"${entry.price_usd:.2f}",
        )
    else:
        return CardPrice(
            name=entry.name,
            price_usd=None,
            price_display="No price",
            error="No price available",
        )


def cache_price(card_name: str, card_price: CardPrice, cache: dict[str, CacheEntry], source: str = "tcgplayer") -> None:
    """Add a card's price to the cache."""
    key = f"{source}:{card_name.strip().lower()}"
    cache[key] = CacheEntry(
        name=card_price.name,
        price_usd=card_price.price_usd,
        fetched_at=datetime.now().isoformat(),
    )


# =============================================================================
# Card Kingdom pricelist handling
# =============================================================================

def load_ck_pricelist() -> dict[str, list[dict]] | None:
    """Load the Card Kingdom pricelist from cache or download fresh."""
    # Check if we have a cached pricelist that's still valid
    if CK_PRICELIST_FILE.exists():
        with open(CK_PRICELIST_FILE, "r") as f:
            data = json.load(f)
        
        fetched_at = datetime.fromisoformat(data.get("fetched_at", "2000-01-01"))
        if datetime.now() - fetched_at < CACHE_MAX_AGE:
            print("📦 Using cached Card Kingdom pricelist")
            return data.get("cards_by_name", {})
    
    # Download fresh pricelist
    print("⬇️  Downloading Card Kingdom pricelist (this may take a moment)...")
    try:
        response = requests.get(
            CARDKINGDOM_API,
            headers={"User-Agent": "MTGCardSorter/1.0"},
            timeout=60,
        )
        
        if response.status_code == 429:
            print("⚠️  Card Kingdom rate limit hit (1 request/hour). Using cached data if available.")
            if CK_PRICELIST_FILE.exists():
                with open(CK_PRICELIST_FILE, "r") as f:
                    data = json.load(f)
                return data.get("cards_by_name", {})
            return None
        
        if response.status_code != 200:
            print(f"⚠️  Failed to fetch Card Kingdom pricelist: {response.status_code}")
            return None
        
        raw_data = response.json()
        cards = raw_data.get("data", [])
        
        # Index cards by name (lowercase) for fast lookup
        cards_by_name: dict[str, list[dict]] = {}
        for card in cards:
            name = card.get("name", "").lower()
            if name:
                if name not in cards_by_name:
                    cards_by_name[name] = []
                cards_by_name[name].append(card)
        
        # Cache the indexed pricelist
        cache_data = {
            "fetched_at": datetime.now().isoformat(),
            "cards_by_name": cards_by_name,
        }
        with open(CK_PRICELIST_FILE, "w") as f:
            json.dump(cache_data, f)
        
        print(f"✅ Downloaded {len(cards)} cards from Card Kingdom")
        return cards_by_name
        
    except requests.RequestException as e:
        print(f"⚠️  Error fetching Card Kingdom pricelist: {e}")
        return None


def lookup_ck_price(card_name: str, pricelist: dict[str, list[dict]]) -> CardPrice:
    """Look up a card's price from the Card Kingdom pricelist."""
    # Normalize name (strip accents) since CK uses ASCII names
    name_key = normalize_name(card_name)
    
    # Try exact match first
    matches = pricelist.get(name_key, [])
    
    # Filter to non-foil, in-stock cards and find cheapest
    in_stock = [
        c for c in matches 
        if c.get("is_foil") == "false" and c.get("qty_retail", 0) > 0
    ]
    
    if not in_stock:
        # Try all non-foil (including out of stock)
        in_stock = [c for c in matches if c.get("is_foil") == "false"]
    
    if not in_stock:
        # Try any version
        in_stock = matches
    
    if not in_stock:
        return CardPrice(
            name=card_name,
            price_usd=None,
            price_display="Not found",
            error="Card not found on Card Kingdom",
        )
    
    # Get cheapest option
    cheapest = min(in_stock, key=lambda c: float(c.get("price_retail", "999999")))
    price = float(cheapest.get("price_retail", 0))
    actual_name = cheapest.get("name", card_name)
    
    return CardPrice(
        name=actual_name,
        price_usd=price,
        price_display=f"${price:.2f}",
    )


def fetch_card_price(card_name: str) -> CardPrice:
    """Fetch the TCGPlayer price for a card via Scryfall API."""
    try:
        response = requests.get(
            SCRYFALL_API,
            params={"exact": card_name.strip()},
            headers={"User-Agent": "MTGCardSorter/1.0"},
            timeout=10,
        )

        if response.status_code == 404:
            # Try fuzzy search for misspellings
            response = requests.get(
                SCRYFALL_API,
                params={"fuzzy": card_name.strip()},
                headers={"User-Agent": "MTGCardSorter/1.0"},
                timeout=10,
            )

        if response.status_code != 200:
            return CardPrice(
                name=card_name,
                price_usd=None,
                price_display="N/A",
                error=f"Card not found: {card_name}",
            )

        data = response.json()
        actual_name = data.get("name", card_name)
        price_usd = data.get("prices", {}).get("usd")

        if price_usd is None:
            # Try foil price if regular is unavailable
            price_usd = data.get("prices", {}).get("usd_foil")

        if price_usd is not None:
            return CardPrice(
                name=actual_name,
                price_usd=float(price_usd),
                price_display=f"${float(price_usd):.2f}",
            )
        else:
            return CardPrice(
                name=actual_name,
                price_usd=None,
                price_display="No price",
                error="No price available",
            )

    except requests.RequestException as e:
        return CardPrice(
            name=card_name,
            price_usd=None,
            price_display="Error",
            error=str(e),
        )


def read_card_list(filepath: str) -> list[str]:
    """Read card names from a text file (one per line)."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Filter empty lines and comments
    cards = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]
    return cards


def fetch_prices(cards: list[str], source: str = "tcgplayer") -> list[CardPrice]:
    """Fetch prices for all cards, using cache when available."""
    results = []
    total = len(cards)
    cache = load_cache()
    cache_hits = 0
    api_calls = 0
    
    source_name = "TCGPlayer" if source == "tcgplayer" else "Card Kingdom"
    print(f"\n🔍 Fetching {source_name} prices for {total} cards...\n")
    
    # For Card Kingdom, load the pricelist once upfront
    ck_pricelist = None
    if source == "cardkingdom":
        ck_pricelist = load_ck_pricelist()
        if ck_pricelist is None:
            print("❌ Failed to load Card Kingdom pricelist")
            sys.exit(1)

    for i, card_name in enumerate(cards, 1):
        print(f"  [{i}/{total}] {card_name}...", end=" ", flush=True)

        # Check cache first
        cached = get_cached_price(card_name, cache, source)
        if cached is not None:
            card_price = cached
            cache_hits += 1
            status = "📦 cached"
        else:
            if source == "cardkingdom":
                card_price = lookup_ck_price(card_name, ck_pricelist)
            else:
                card_price = fetch_card_price(card_name)
                # Be respectful to Scryfall's servers (only delay for API calls)
                if i < total:
                    time.sleep(REQUEST_DELAY)
            
            cache_price(card_name, card_price, cache, source)
            api_calls += 1
            status = "✓"

        results.append(card_price)

        if card_price.error:
            print(f"⚠️  {card_price.error}")
        else:
            print(f"{status} {card_price.price_display}")

    # Save updated cache
    save_cache(cache)
    if source == "cardkingdom":
        print(f"\n  💾 Cache: {cache_hits} hits, {api_calls} lookups")
    else:
        print(f"\n  💾 Cache: {cache_hits} hits, {api_calls} API calls")

    return results


def sort_cards_by_price(
    results: list[CardPrice], descending: bool = False
) -> list[CardPrice]:
    """Sort card prices by price."""
    # Sort: cards with prices first, then cards without prices
    priced = [r for r in results if r.price_usd is not None]
    unpriced = [r for r in results if r.price_usd is None]

    priced.sort(key=lambda x: x.price_usd, reverse=descending)

    return priced + unpriced


def filter_by_threshold(
    results: list[CardPrice], threshold: float
) -> tuple[list[CardPrice], list[CardPrice]]:
    """Split cards into two lists: below threshold and at/above threshold."""
    below = []
    above = []

    for card in results:
        if card.price_usd is None:
            above.append(card)  # Unpriced cards go in the "above" pile (need manual check)
        elif card.price_usd < threshold:
            below.append(card)
        else:
            above.append(card)

    # Sort each list by price
    below.sort(key=lambda x: x.price_usd)

    above_priced = [c for c in above if c.price_usd is not None]
    above_unpriced = [c for c in above if c.price_usd is None]
    above_priced.sort(key=lambda x: x.price_usd)

    return below, above_priced + above_unpriced


def display_results(results: list[CardPrice], descending: bool) -> None:
    """Display the sorted results in a nice format."""
    order = "highest to lowest" if descending else "lowest to highest"
    print(f"\n{'='*60}")
    print(f"📊 Cards sorted by price ({order}):")
    print(f"{'='*60}\n")

    total_value = 0.0
    priced_count = 0

    for i, card in enumerate(results, 1):
        if card.price_usd is not None:
            print(f"  {i:3}. {card.price_display:>10}  {card.name}")
            total_value += card.price_usd
            priced_count += 1
        else:
            print(f"  {i:3}. {'N/A':>10}  {card.name} ({card.error})")

    print(f"\n{'='*60}")
    print(f"💰 Total estimated value: ${total_value:.2f} ({priced_count} cards priced)")
    print(f"{'='*60}\n")


def tcgplayer_name(name: str) -> str:
    """Convert card name to TCGPlayer format (front face only for DFCs)."""
    # Double-faced cards come as "Front // Back" but TCGPlayer wants just "Front"
    if " // " in name:
        return name.split(" // ")[0]
    return name


def save_results(results: list[CardPrice], filepath: str) -> None:
    """Save the card names to a file in TCGPlayer format (1 Card Name)."""
    with open(filepath, "w") as f:
        for card in results:
            f.write(f"1 {tcgplayer_name(card.name)}\n")

    print(f"✅ Results saved to: {filepath}")


def save_summary(
    below: list[CardPrice],
    above: list[CardPrice],
    threshold: float,
    filepath: str,
    source: str = "tcgplayer",
) -> None:
    """Save a summary file with costs and price-sorted card list."""
    def calc_total(cards: list[CardPrice]) -> tuple[float, int]:
        total = sum(c.price_usd for c in cards if c.price_usd is not None)
        count = sum(1 for c in cards if c.price_usd is not None)
        return total, count

    below_total, below_count = calc_total(below)
    above_total, above_count = calc_total(above)
    
    if source == "cardkingdom":
        source_display = "Card Kingdom (via cardkingdom.com API)"
    else:
        source_display = "TCGPlayer (via Scryfall API)"

    with open(filepath, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"MTG Card Price Summary (Threshold: ${threshold:.2f})\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Price Source:  {source_display}\n")
        f.write(f"Retrieved:     {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write(f"BELOW ${threshold:.2f}:\n")
        f.write(f"  Cards:    {below_count}\n")
        f.write(f"  Total:    ${below_total:.2f}\n\n")

        f.write(f"AT OR ABOVE ${threshold:.2f}:\n")
        f.write(f"  Cards:    {above_count}\n")
        f.write(f"  Total:    ${above_total:.2f}\n\n")

        f.write(f"GRAND TOTAL: ${below_total + above_total:.2f} ({below_count + above_count} cards)\n\n")

        f.write("=" * 60 + "\n")
        f.write(f"Cards BELOW ${threshold:.2f} (sorted by price)\n")
        f.write("=" * 60 + "\n\n")

        for card in below:
            if card.price_usd is not None:
                f.write(f"{card.price_display:>10} | {card.name}\n")
            else:
                f.write(f"{'N/A':>10} | {card.name}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Cards AT OR ABOVE ${threshold:.2f} (sorted by price)\n")
        f.write("=" * 60 + "\n\n")

        for card in above:
            f.write(f"{card.price_display:>10} | {card.name}\n")

    print(f"✅ Summary saved to: {filepath}")


def display_filtered_results(
    below: list[CardPrice],
    above: list[CardPrice],
    threshold: float,
    source: str = "tcgplayer",
) -> None:
    """Display the filtered results split by threshold."""
    def calc_total(cards: list[CardPrice]) -> tuple[float, int]:
        total = sum(c.price_usd for c in cards if c.price_usd is not None)
        count = sum(1 for c in cards if c.price_usd is not None)
        return total, count

    below_total, below_count = calc_total(below)
    above_total, above_count = calc_total(above)
    
    source_name = "Card Kingdom" if source == "cardkingdom" else "TCGPlayer"

    print(f"\n{'='*60}")
    print(f"📊 Cards BELOW ${threshold:.2f} ({len(below)} cards):")
    print(f"{'='*60}\n")

    for i, card in enumerate(below, 1):
        if card.price_usd is not None:
            print(f"  {i:3}. {card.price_display:>10}  {card.name}")
        else:
            print(f"  {i:3}. {'N/A':>10}  {card.name} ({card.error})")

    print(f"\n{'='*60}")
    print(f"📊 Cards AT OR ABOVE ${threshold:.2f} ({len(above)} cards):")
    print(f"{'='*60}\n")

    for i, card in enumerate(above, 1):
        print(f"  {i:3}. {card.price_display:>10}  {card.name}")

    print(f"\n  💵 Subtotal: ${above_total:.2f} ({above_count} cards)")

    print(f"\n{'='*60}")
    print(f"🛒 {source_name} Estimate for BELOW ${threshold:.2f} list:")
    print(f"{'='*60}")
    print(f"\n   Cards:    {below_count}")
    print(f"   Subtotal: ${below_total:.2f}")
    print(f"   ─────────────────────")
    print(f"   TOTAL:    ${below_total:.2f}  (+ shipping)")
    print()
    print(f"💰 Grand total (all cards): ${below_total + above_total:.2f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Sort MTG cards by price using TCGPlayer or Card Kingdom"
    )
    parser.add_argument("input_file", help="Text file with card names (one per line)")
    parser.add_argument(
        "--source", "-s",
        choices=["tcgplayer", "cardkingdom"],
        default="tcgplayer",
        help="Price source: tcgplayer (default) or cardkingdom",
    )
    parser.add_argument(
        "--descending", "-d",
        action="store_true",
        help="Sort from highest to lowest price (default: lowest to highest)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (optional). With --threshold, creates two files: <name>_below.txt and <name>_above.txt",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        help="Price threshold to split cards (e.g., 1.00 for $1.00). Creates two lists: below and at/above threshold.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the price cache before fetching",
    )

    args = parser.parse_args()

    # Handle cache clearing (keeps CK pricelist since it's rate-limited)
    if args.clear_cache:
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        print("🗑️  Price cache cleared (Card Kingdom pricelist preserved)")

    try:
        cards = read_card_list(args.input_file)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {args.input_file}")
        sys.exit(1)

    if not cards:
        print("❌ Error: No cards found in the input file")
        sys.exit(1)

    print(f"📋 Loaded {len(cards)} cards from {args.input_file}")

    # Fetch all prices (with caching)
    results = fetch_prices(cards, source=args.source)

    if args.threshold is not None:
        # Filter mode: split into two lists
        below, above = filter_by_threshold(results, args.threshold)
        display_filtered_results(below, above, args.threshold, source=args.source)

        if args.output:
            # Generate output files
            base = Path(args.output)
            stem = base.stem
            suffix = base.suffix or ".txt"
            parent = base.parent

            below_file = parent / f"{stem}_below{suffix}"
            above_file = parent / f"{stem}_above{suffix}"
            summary_file = parent / f"{stem}_summary{suffix}"

            save_results(below, str(below_file))
            save_results(above, str(above_file))
            save_summary(below, above, args.threshold, str(summary_file), source=args.source)
    else:
        # Sort mode: single sorted list
        sorted_results = sort_cards_by_price(results, descending=args.descending)
        display_results(sorted_results, descending=args.descending)

        if args.output:
            save_results(sorted_results, args.output)


if __name__ == "__main__":
    main()
