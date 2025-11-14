#!/usr/bin/env python3
"""
Test Script for Schema Compliance

Validates that generated transactions match the schema defined in config/schema.txt
"""

from src.enhanced_fraud_generator import EnhancedFraudGenerator
import pandas as pd

# Expected schema fields from config/schema.txt
EXPECTED_FIELDS = [
    'payment_rail',
    'transaction_id_1',
    'transaction_id_2',
    'sending_fedline_tier',
    'odfi',
    'odfi_name',
    'sending_net_source',
    'instructing_fi_id_code',
    'instructing_fi_id',
    'instructing_fi_name',
    'ofi_id_code',
    'ofi_id',
    'ofi_name',
    'originating_account_type',
    'originating_account_identifier',
    'originating_account_description',
    'rdfi',
    'rdfi_name',
    'receiving_net_source',
    'receiving_fedline_tier',
    'intermediary_fi_id_code',
    'intermediary_fi_id',
    'intermediary_fi_name',
    'rfi_id_code',
    'rfi_id',
    'rfi_name',
    'receiving_account_type',
    'receiving_account_identifier',
    'receiving_account_description',
    'pay_type',
    'pay_type_subtype_1',
    'pay_type_subtype_2',
    'return_/_error_reason',
    'pay_code',
    'pay_code_type',
    'originator_type',
    'transaction_purpose',
    'value_type',
    'reference_id',
    'previous_message_id',
    'payment_speed_category',
    'transaction_create_date',
    'transaction_processed_date',
    'odfi_transaction_settlement_date',
    'rdfi_transaction_settlement_date',
    'dollar_value',
    'security_par_value',
    'dollar_direction',
    'security_direction',
    'addenda_count',
    'recurring_payment'
]

def test_schema_compliance():
    """Test that generated transactions comply with schema"""
    print("=" * 70)
    print("TESTING SCHEMA COMPLIANCE")
    print("=" * 70)
    print()

    # Generate small test dataset
    print("1. Generating test dataset (small scale)...")
    generator = EnhancedFraudGenerator('config/enhanced_fraud_rules.yaml')

    # Temporarily reduce transaction count for faster testing
    original_count = generator.config['transaction_generation']['total_transactions']
    generator.config['transaction_generation']['total_transactions'] = 100

    data = generator.generate_network()
    transactions_df = data['transactions']

    print(f"   ✓ Generated {len(transactions_df)} transactions")
    print()

    # Check schema compliance
    print("2. Validating schema compliance...")
    print()

    # Get actual fields
    actual_fields = set(transactions_df.columns)
    expected_fields = set(EXPECTED_FIELDS)

    # Check for missing fields
    missing_fields = expected_fields - actual_fields
    if missing_fields:
        print(f"   ✗ MISSING FIELDS ({len(missing_fields)}):")
        for field in sorted(missing_fields):
            print(f"     - {field}")
        print()
    else:
        print("   ✓ All expected schema fields present!")
        print()

    # Check for extra fields
    extra_fields = actual_fields - expected_fields
    if extra_fields:
        print(f"   ℹ️  EXTRA FIELDS ({len(extra_fields)}) - may be OK:")
        for field in sorted(extra_fields):
            print(f"     + {field}")
        print()

    # Summary
    print("3. Schema Validation Summary")
    print(f"   Expected fields: {len(expected_fields)}")
    print(f"   Actual fields: {len(actual_fields)}")
    print(f"   Missing: {len(missing_fields)}")
    print(f"   Extra: {len(extra_fields)}")
    print()

    # Sample data validation
    print("4. Sample Transaction (First Row):")
    print("-" * 70)
    sample_tx = transactions_df.iloc[0]

    # Print key fields
    key_fields = [
        'payment_rail', 'transaction_id_1', 'dollar_value',
        'odfi_name', 'rdfi_name', 'payment_speed_category',
        'transaction_create_date', 'recurring_payment'
    ]

    for field in key_fields:
        if field in sample_tx:
            print(f"   {field:35s}: {sample_tx[field]}")

    print()

    # Data type validation
    print("5. Data Type Validation:")
    print("-" * 70)

    # Check timestamp fields
    timestamp_fields = [
        'transaction_create_date',
        'transaction_processed_date',
        'odfi_transaction_settlement_date',
        'rdfi_transaction_settlement_date'
    ]

    for field in timestamp_fields:
        if field in transactions_df.columns:
            # Try to parse as datetime
            try:
                pd.to_datetime(transactions_df[field].iloc[0])
                print(f"   ✓ {field:40s}: Valid timestamp")
            except:
                print(f"   ✗ {field:40s}: Invalid timestamp format")

    # Check numeric fields
    numeric_fields = ['dollar_value', 'security_par_value', 'addenda_count']
    for field in numeric_fields:
        if field in transactions_df.columns:
            if pd.api.types.is_numeric_dtype(transactions_df[field]):
                print(f"   ✓ {field:40s}: Numeric type")
            else:
                print(f"   ⚠️  {field:40s}: Not numeric")

    print()

    # Payment rail distribution
    print("6. Payment Rail Distribution:")
    print("-" * 70)
    if 'payment_rail' in transactions_df.columns:
        rail_counts = transactions_df['payment_rail'].value_counts()
        for rail, count in rail_counts.items():
            pct = (count / len(transactions_df)) * 100
            print(f"   {rail:10s}: {count:3d} ({pct:5.1f}%)")
    print()

    # Final assessment
    print("=" * 70)
    if len(missing_fields) == 0:
        print("✅ SCHEMA COMPLIANCE: PASSED")
        print("   All required fields are present in generated data.")
    else:
        print("❌ SCHEMA COMPLIANCE: FAILED")
        print(f"   {len(missing_fields)} required fields are missing.")

    print("=" * 70)
    print()

    return len(missing_fields) == 0


if __name__ == '__main__':
    success = test_schema_compliance()
    exit(0 if success else 1)
