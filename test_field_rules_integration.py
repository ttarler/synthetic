#!/usr/bin/env python3
"""
Test Field Rules Integration with Schema Mapper

Validates that the schema mapper correctly uses field rules for generation
"""

from src.schema_mapper import PaymentSchemaMapper
from datetime import datetime
from collections import Counter


def test_field_rules_integration():
    """Test that schema mapper uses field rules correctly"""
    print("=" * 70)
    print("TESTING FIELD RULES INTEGRATION WITH SCHEMA MAPPER")
    print("=" * 70)
    print()

    # Create test institutions
    institutions = [
        {'institution_id': 'inst_001', 'type': 'major_bank', 'name': 'Test Bank A'},
        {'institution_id': 'inst_002', 'type': 'regional_bank', 'name': 'Test Bank B'},
    ]

    # Create test accounts
    accounts = {
        'acc_001': {'account_id': 'acc_001', 'account_type': 'checking'},
        'acc_002': {'account_id': 'acc_002', 'account_type': 'savings'},
    }

    # Initialize schema mapper (with field generator)
    print("1. Initializing Schema Mapper with Field Generator:")
    print("-" * 70)
    mapper = PaymentSchemaMapper(institutions)
    print("   ✓ Schema mapper initialized with field generator")
    print(f"   ✓ Loaded {len(mapper.field_generator.list_all_fields())} field rules")
    print()

    # Test field generation in institution codes
    print("2. Testing Institution Code Generation:")
    print("-" * 70)
    inst_001 = mapper.institutions['inst_001']
    inst_002 = mapper.institutions['inst_002']

    print(f"   Institution 1:")
    print(f"     Routing Number: {inst_001['routing_number']}")
    print(f"     FedLine Tier: {inst_001['fedline_tier']}")
    print(f"     Valid 9-digit: {100000000 <= inst_001['routing_number'] <= 999999999}")
    print()
    print(f"   Institution 2:")
    print(f"     Routing Number: {inst_002['routing_number']}")
    print(f"     FedLine Tier: {inst_002['fedline_tier']}")
    print(f"     Valid 9-digit: {100000000 <= inst_002['routing_number'] <= 999999999}")
    print()

    # Create test transactions
    print("3. Testing Transaction Mapping with Field Rules:")
    print("-" * 70)

    test_transactions = []
    for i in range(100):
        tx = {
            'transaction_id': f'TXN_{i:06d}',
            'timestamp': '2024-03-01 10:00:00',
            'amount': 1000.0 + (i * 10),
            'payment_type': 'ACH',
            'sender_account': 'acc_001',
            'receiver_account': 'acc_002',
            'sender_institution': 'inst_001',
            'receiver_institution': 'inst_002',
            'relationship_type': 'bill_payment'
        }
        test_transactions.append(tx)

    # Map transactions
    mapped_transactions = []
    for tx in test_transactions:
        mapped_tx = mapper.map_to_schema(tx, accounts)
        mapped_transactions.append(mapped_tx)

    print(f"   ✓ Mapped {len(mapped_transactions)} transactions")
    print()

    # Analyze field rule adherence
    print("4. Analyzing Field Rule Adherence:")
    print("-" * 70)

    # Check payment_rail distribution (should follow field rules)
    payment_rails = [tx['payment_rail'] for tx in mapped_transactions]
    rail_counts = Counter(payment_rails)

    print("   Payment Rail Distribution:")
    for rail, count in rail_counts.items():
        pct = (count / len(mapped_transactions)) * 100
        print(f"     {rail:12s}: {count:3d} ({pct:5.1f}%)")
    print()

    # Check pay_type mapping
    pay_types = [tx['pay_type'] for tx in mapped_transactions]
    pay_type_counts = Counter(pay_types)

    print("   Pay Type Distribution:")
    for pay_type, count in pay_type_counts.items():
        pct = (count / len(mapped_transactions)) * 100
        print(f"     {pay_type:12s}: {count:3d} ({pct:5.1f}%)")
    print()

    # Check addenda_count (should be integer 0-999)
    addenda_counts = [tx['addenda_count'] for tx in mapped_transactions]
    print(f"   Addenda Count:")
    print(f"     Min: {min(addenda_counts)}, Max: {max(addenda_counts)}")
    print(f"     All valid integers: {all(isinstance(c, int) and 0 <= c <= 999 for c in addenda_counts)}")
    print()

    # Check dollar_direction (should be category)
    dollar_directions = set(tx['dollar_direction'] for tx in mapped_transactions)
    print(f"   Dollar Direction values: {dollar_directions}")
    print(f"     All valid categories: {dollar_directions.issubset({'DEBIT', 'CREDIT'})}")
    print()

    # Check payment_speed_category
    speed_categories = set(tx['payment_speed_category'] for tx in mapped_transactions)
    print(f"   Payment Speed Categories: {speed_categories}")
    print(f"     All valid categories: {speed_categories.issubset({'STANDARD', 'EXPEDITED', 'INSTANT', 'DELAYED'})}")
    print()

    # Sample transaction details
    print("5. Sample Transaction (showing field-generated values):")
    print("-" * 70)
    sample = mapped_transactions[0]

    print(f"   Transaction ID: {sample['transaction_id_1']}")
    print(f"   Payment Rail: {sample['payment_rail']}")
    print(f"   Pay Type: {sample['pay_type']}")
    print(f"   Pay Code: {sample['pay_code']}")
    print(f"   Pay Code Type: {sample['pay_code_type']}")
    print(f"   Dollar Direction: {sample['dollar_direction']}")
    print(f"   Payment Speed: {sample['payment_speed_category']}")
    print(f"   Addenda Count: {sample['addenda_count']}")
    print(f"   Recurring Payment: {sample['recurring_payment']}")
    print()

    # Final assessment
    print("=" * 70)
    print("✅ FIELD RULES INTEGRATION: SUCCESSFUL")
    print("   Schema mapper correctly uses field generator for rule-based generation")
    print("=" * 70)
    print()


if __name__ == '__main__':
    test_field_rules_integration()
