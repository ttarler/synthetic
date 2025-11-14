#!/usr/bin/env python3
"""
Test Script for Field Generator

Validates field generation with:
- Category fields with distributions
- Integer fields with various distributions
- Double fields with various distributions
- Regex-based string generation
- Field validation
"""

from src.field_generator import FieldGenerator
import numpy as np
from collections import Counter


def test_category_generation():
    """Test category field generation with distributions"""
    print("=" * 70)
    print("TESTING CATEGORY FIELD GENERATION")
    print("=" * 70)
    print()

    generator = FieldGenerator()

    # Test payment_rail field
    print("1. Testing payment_rail field (category with distribution):")
    print("-" * 70)

    values = [generator.generate_value('payment_rail') for _ in range(1000)]
    counts = Counter(values)

    print(f"   Generated 1000 values:")
    for value, count in counts.most_common():
        pct = (count / 1000) * 100
        print(f"     {value:15s}: {count:4d} ({pct:5.1f}%)")

    # Check that distribution roughly matches expected
    expected = {'ACH': 60, 'CHECK': 15, 'FUNDS': 15, 'SECURITIES': 5, 'CASH': 5}
    print(f"\n   Expected distribution (%):")
    for value, exp_pct in expected.items():
        actual_pct = (counts[value] / 1000) * 100
        diff = abs(actual_pct - exp_pct)
        status = "✓" if diff < 5 else "⚠️"
        print(f"     {status} {value:15s}: {exp_pct}% (actual: {actual_pct:.1f}%)")
    print()


def test_integer_distributions():
    """Test integer generation with different distributions"""
    print("=" * 70)
    print("TESTING INTEGER FIELD DISTRIBUTIONS")
    print("=" * 70)
    print()

    generator = FieldGenerator()

    # Test uniform distribution (default)
    print("1. Testing ODFI field (9-digit integer, uniform distribution):")
    print("-" * 70)
    values = [generator.generate_value('odfi') for _ in range(100)]
    print(f"   Sample values: {values[:5]}")
    print(f"   Min: {min(values)}, Max: {max(values)}")
    print(f"   All 9 digits: {all(100000000 <= v <= 999999999 for v in values)}")
    print()

    # Test with custom distribution
    print("2. Testing custom normal distribution:")
    print("-" * 70)
    custom_rule = {
        'type': 'integer',
        'min': 0,
        'max': 100,
        'distribution_type': 'normal',
        'mean': 50,
        'std': 15
    }

    generator.field_rules['test_normal'] = custom_rule
    values = [generator.generate_value('test_normal') for _ in range(1000)]

    print(f"   Generated 1000 values (normal: mean=50, std=15):")
    print(f"   Mean: {np.mean(values):.2f} (expected ~50)")
    print(f"   Std:  {np.std(values):.2f} (expected ~15)")
    print(f"   Min:  {min(values)}, Max: {max(values)}")

    # Check if roughly 68% are within 1 std of mean
    within_1std = sum(1 for v in values if 35 <= v <= 65)
    pct_1std = (within_1std / 1000) * 100
    print(f"   Within 1 std: {pct_1std:.1f}% (expected ~68%)")
    print()

    # Test Poisson distribution
    print("3. Testing custom Poisson distribution:")
    print("-" * 70)
    custom_rule = {
        'type': 'integer',
        'min': 0,
        'max': 50,
        'distribution_type': 'poisson',
        'lambda': 10
    }

    generator.field_rules['test_poisson'] = custom_rule
    values = [generator.generate_value('test_poisson') for _ in range(1000)]

    print(f"   Generated 1000 values (Poisson: λ=10):")
    print(f"   Mean: {np.mean(values):.2f} (expected ~10)")
    print(f"   Std:  {np.std(values):.2f} (expected ~√10 ≈ 3.16)")
    print(f"   Min:  {min(values)}, Max: {max(values)}")
    print()


def test_double_distributions():
    """Test double/float generation with different distributions"""
    print("=" * 70)
    print("TESTING DOUBLE FIELD DISTRIBUTIONS")
    print("=" * 70)
    print()

    generator = FieldGenerator()

    # Test uniform distribution (default)
    print("1. Testing dollar_value field (uniform distribution):")
    print("-" * 70)
    values = [generator.generate_value('dollar_value') for _ in range(100)]
    print(f"   Sample values: {[f'${v:,.2f}' for v in values[:5]]}")
    print(f"   Min: ${min(values):,.2f}, Max: ${max(values):,.2f}")
    print(f"   All positive: {all(v > 0 for v in values)}")
    print()

    # Test lognormal distribution
    print("2. Testing custom lognormal distribution:")
    print("-" * 70)
    custom_rule = {
        'type': 'double',
        'min': 0.01,
        'max': 100000.0,
        'precision': 2,
        'distribution_type': 'lognormal',
        'mean': 6.9,  # ln(1000)
        'sigma': 1.0
    }

    generator.field_rules['test_lognormal'] = custom_rule
    values = [generator.generate_value('test_lognormal') for _ in range(1000)]

    print(f"   Generated 1000 values (lognormal: mean=6.9, sigma=1.0):")
    print(f"   Median: ${np.median(values):,.2f} (expected ~${np.exp(6.9):,.2f})")
    print(f"   Mean:   ${np.mean(values):,.2f}")
    print(f"   Min:    ${min(values):,.2f}, Max: ${max(values):,.2f}")

    # Show distribution of values
    ranges = [(0, 100), (100, 500), (500, 2000), (2000, 10000), (10000, 100000)]
    print(f"   Distribution:")
    for low, high in ranges:
        count = sum(1 for v in values if low <= v < high)
        pct = (count / 1000) * 100
        print(f"     ${low:6,} - ${high:6,}: {count:4d} ({pct:5.1f}%)")
    print()

    # Test beta distribution
    print("3. Testing custom beta distribution:")
    print("-" * 70)
    custom_rule = {
        'type': 'double',
        'min': 0.0,
        'max': 1.0,
        'precision': 4,
        'distribution_type': 'beta',
        'alpha': 8.0,
        'beta': 2.0
    }

    generator.field_rules['test_beta'] = custom_rule
    values = [generator.generate_value('test_beta') for _ in range(1000)]

    print(f"   Generated 1000 values (beta: α=8.0, β=2.0):")
    print(f"   Mean: {np.mean(values):.4f} (expected ~{8/(8+2):.4f})")
    print(f"   Min:  {min(values):.4f}, Max: {max(values):.4f}")

    # Show how many are above 0.5
    above_half = sum(1 for v in values if v > 0.5)
    pct_above = (above_half / 1000) * 100
    print(f"   Above 0.5: {pct_above:.1f}% (expected >80% for α=8, β=2)")
    print()


def test_string_generation():
    """Test string generation including regex patterns"""
    print("=" * 70)
    print("TESTING STRING FIELD GENERATION")
    print("=" * 70)
    print()

    generator = FieldGenerator()

    # Test format-based generation
    print("1. Testing transaction_id_1 field (format-based):")
    print("-" * 70)
    values = [generator.generate_value('transaction_id_1', {'_seq_num': i}) for i in range(5)]
    print(f"   Sample values: {values}")
    print()

    # Test regex-based generation
    print("2. Testing regex-based string generation:")
    print("-" * 70)

    # Custom field with regex pattern
    custom_rule = {
        'type': 'string',
        'pattern': r'^[A-Z]{3}-\d{6}$',
        'generate_from_pattern': True,
        'description': 'Custom ID like ABC-123456'
    }

    generator.field_rules['test_regex'] = custom_rule
    values = [generator.generate_value('test_regex') for _ in range(10)]

    print(f"   Pattern: ^[A-Z]{{3}}-\\d{{6}}$")
    print(f"   Sample values:")
    for v in values:
        # Validate against pattern
        import re
        matches = bool(re.match(r'^[A-Z]{3}-\d{6}$', v))
        status = "✓" if matches else "✗"
        print(f"     {status} {v}")
    print()

    # Test intermediary routing number pattern
    print("3. Testing intermediary_fi_id field (Fed routing pattern):")
    print("-" * 70)

    # This field uses pattern validation but not generation in current config
    # Let's create a custom rule that generates from pattern
    custom_rule = {
        'type': 'string',
        'pattern': r'^0\d{8}$',
        'generate_from_pattern': True,
        'description': 'Fed routing number starting with 0'
    }

    generator.field_rules['test_fed_routing'] = custom_rule
    values = [generator.generate_value('test_fed_routing') for _ in range(10)]

    print(f"   Pattern: ^0\\d{{8}}$")
    print(f"   Sample values:")
    for v in values:
        import re
        matches = bool(re.match(r'^0\d{8}$', v))
        status = "✓" if matches else "✗"
        print(f"     {status} {v}")
    print()


def test_validation():
    """Test field validation"""
    print("=" * 70)
    print("TESTING FIELD VALIDATION")
    print("=" * 70)
    print()

    generator = FieldGenerator()

    # Test valid values
    print("1. Testing valid field values:")
    print("-" * 70)

    test_cases = [
        ('payment_rail', 'ACH', True),
        ('payment_rail', 'INVALID', False),
        ('odfi', 123456789, True),
        ('odfi', 12345, False),  # Too small
        ('dollar_value', 100.50, True),
        ('dollar_value', -10.0, False),  # Negative
    ]

    for field, value, expected in test_cases:
        is_valid, error_msg = generator.validate_value(field, value)
        status = "✓" if is_valid == expected else "✗"
        result = "PASS" if is_valid else f"FAIL: {error_msg}"
        print(f"   {status} {field} = {value}: {result}")
    print()


def test_derived_fields():
    """Test derived field generation"""
    print("=" * 70)
    print("TESTING DERIVED FIELD GENERATION")
    print("=" * 70)
    print()

    generator = FieldGenerator()

    print("1. Testing transaction_id_2 (derived from transaction_id_1):")
    print("-" * 70)

    context = {'transaction_id_1': 'TXN_000123'}
    tx_id_2 = generator.generate_value('transaction_id_2', context)
    print(f"   transaction_id_1: {context['transaction_id_1']}")
    print(f"   transaction_id_2: {tx_id_2}")
    print(f"   Derived correctly: {tx_id_2 == context['transaction_id_1'] + '_REF'}")
    print()

    print("2. Testing pay_type (mapped from payment_rail):")
    print("-" * 70)

    test_mappings = [
        ('ACH', 'ACH'),
        ('FUNDS', 'RTP'),
        ('SECURITIES', 'SECURITIES'),
        ('CHECK', 'CHECK'),
        ('CASH', 'CASH'),
    ]

    for payment_rail, expected_pay_type in test_mappings:
        context = {'payment_rail': payment_rail}
        pay_type = generator.generate_value('pay_type', context)
        status = "✓" if pay_type == expected_pay_type else "✗"
        print(f"   {status} payment_rail={payment_rail:12s} → pay_type={pay_type} (expected: {expected_pay_type})")
    print()


def test_dependencies():
    """Test cross-field dependencies"""
    print("=" * 70)
    print("TESTING CROSS-FIELD DEPENDENCIES")
    print("=" * 70)
    print()

    generator = FieldGenerator()

    print("1. Testing SECURITIES payment_rail dependencies:")
    print("-" * 70)

    field_values = {
        'payment_rail': 'SECURITIES',
        'dollar_value': 10000.0
    }

    # Apply dependencies
    updated_values = generator.apply_dependencies(field_values)

    print(f"   Input: payment_rail = {field_values['payment_rail']}")
    print(f"   After applying dependencies:")
    print(f"     security_par_value: {updated_values.get('security_par_value', 'N/A')}")
    print(f"     security_direction: {updated_values.get('security_direction', 'N/A')}")
    print(f"     value_type: {updated_values.get('value_type', 'N/A')}")
    print()


if __name__ == '__main__':
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "FIELD GENERATOR TEST SUITE" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")

    test_category_generation()
    test_integer_distributions()
    test_double_distributions()
    test_string_generation()
    test_validation()
    test_derived_fields()
    test_dependencies()

    print("=" * 70)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 70)
    print()
