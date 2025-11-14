#!/usr/bin/env python3
"""
Test Script for Institutional Relationship Matrix

Validates that ODFI-RDFI patterns are being used correctly in transaction generation
"""

from src.enhanced_fraud_generator import EnhancedFraudGenerator
import pandas as pd
from collections import Counter

def test_institutional_relationships():
    """Test that institutional relationships are properly used"""
    print("=" * 70)
    print("TESTING INSTITUTIONAL RELATIONSHIP MATRIX")
    print("=" * 70)
    print()

    # Generate test dataset with institutional relationships
    print("1. Generating test dataset with institutional relationships...")
    generator = EnhancedFraudGenerator('config/enhanced_fraud_rules.yaml')

    # Temporarily reduce transaction count for faster testing
    generator.config['transaction_generation']['total_transactions'] = 200

    data = generator.generate_network()
    transactions_df = data['transactions']
    institutions_df = data['institutions']

    print(f"   ✓ Generated {len(transactions_df)} transactions")
    print(f"   ✓ Using {len(institutions_df)} institutions")
    print()

    # Check if institutional relationships were loaded
    print("2. Institutional Relationship Matrix Status:")
    print("-" * 70)
    if generator.institutional_relationships and generator.institutional_relationships.is_loaded:
        print("   ✓ Matrix loaded successfully")
        stats = generator.institutional_relationships.get_statistics()
        print(f"   ✓ Matrix has {stats['total_relationships']} institutional relationships")
        print(f"   ✓ Covering {stats['unique_institutions']} unique institutions")
        print(f"   ✓ Avg {stats['avg_relationships_per_institution']:.1f} relationships per institution")
        print()
    else:
        print("   ✗ Matrix NOT loaded - using random selection")
        print()
        return False

    # Analyze ODFI-RDFI patterns in generated transactions
    print("3. Analyzing Generated Transaction Patterns:")
    print("-" * 70)

    # Get ODFI-RDFI pairs from generated transactions
    odfi_rdfi_pairs = list(zip(transactions_df['odfi'], transactions_df['rdfi']))
    pair_counts = Counter(odfi_rdfi_pairs)

    print(f"   Total unique ODFI-RDFI pairs: {len(pair_counts)}")
    print(f"   Most common pairs:")
    for (odfi, rdfi), count in pair_counts.most_common(5):
        pct = (count / len(transactions_df)) * 100
        print(f"     {odfi} → {rdfi}: {count} transactions ({pct:.1f}%)")
    print()

    # Check for concentration (should be higher with matrix)
    top_10_pct = sum(count for _, count in pair_counts.most_common(10)) / len(transactions_df) * 100
    print(f"   Top 10 pairs represent: {top_10_pct:.1f}% of transactions")
    if top_10_pct > 30:
        print("   ✓ Good concentration - matrix patterns being followed")
    else:
        print("   ⚠️  Low concentration - may indicate random selection")
    print()

    # Compare with matrix data
    print("4. Comparing with Matrix Data:")
    print("-" * 70)

    matrix_df = generator.institutional_relationships.matrix_df
    # Normalize routing numbers to strings without decimals for comparison
    def normalize_routing(r):
        """Normalize routing number to comparable format"""
        s = str(r)
        # Remove .0 suffix if present
        if s.endswith('.0'):
            s = s[:-2]
        return s

    matrix_pairs = set(zip(
        matrix_df['odfi'].apply(normalize_routing),
        matrix_df['rdfi'].apply(normalize_routing)
    ))
    generated_pairs = set((normalize_routing(odfi), normalize_routing(rdfi)) for odfi, rdfi in odfi_rdfi_pairs)

    pairs_in_matrix = generated_pairs.intersection(matrix_pairs)
    coverage_pct = len(pairs_in_matrix) / len(generated_pairs) * 100 if generated_pairs else 0

    print(f"   Matrix contains {len(matrix_pairs)} ODFI-RDFI relationships")
    print(f"   Generated data has {len(generated_pairs)} unique pairs")
    print(f"   {len(pairs_in_matrix)} pairs match matrix ({coverage_pct:.1f}%)")

    if coverage_pct > 70:
        print("   ✓ High coverage - matrix is being used effectively")
    elif coverage_pct > 40:
        print("   ~ Moderate coverage - matrix partially used")
    else:
        print("   ⚠️  Low coverage - matrix may not be applied correctly")
    print()

    # Check payment rail distribution matches matrix patterns
    print("5. Payment Rail Distribution:")
    print("-" * 70)
    rail_dist = transactions_df['payment_rail'].value_counts(normalize=True) * 100
    for rail, pct in rail_dist.items():
        print(f"   {rail:10s}: {pct:5.1f}%")
    print()

    # Institution volume distribution
    print("6. Institution Volume Distribution:")
    print("-" * 70)

    odfi_volumes = transactions_df.groupby('odfi').size().sort_values(ascending=False)
    print(f"   Most active ODFIs (sending institutions):")
    for odfi, count in odfi_volumes.head(5).items():
        pct = (count / len(transactions_df)) * 100
        # Find institution name
        inst_name = "Unknown"
        for inst in generator.institutions:
            if inst.get('routing_number') == odfi:
                inst_name = inst['name']
                break
        print(f"     {odfi} ({inst_name}): {count} transactions ({pct:.1f}%)")
    print()

    # Final assessment
    print("=" * 70)
    if coverage_pct > 70 and top_10_pct > 30:
        print("✅ INSTITUTIONAL RELATIONSHIPS: WORKING CORRECTLY")
        print("   Matrix patterns are being followed in transaction generation")
    elif coverage_pct > 40:
        print("⚠️  INSTITUTIONAL RELATIONSHIPS: PARTIALLY WORKING")
        print("   Matrix is being used but coverage could be improved")
    else:
        print("❌ INSTITUTIONAL RELATIONSHIPS: NOT WORKING")
        print("   Generated transactions don't follow matrix patterns")

    print("=" * 70)
    print()

    return coverage_pct > 70


if __name__ == '__main__':
    success = test_institutional_relationships()
    exit(0 if success else 1)
