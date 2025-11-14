# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **fraud detection workshop/demo** that generates synthetic financial network data with sophisticated fraud patterns. The project uses AWS services (Neptune, SageMaker, QuickSight) to demonstrate graph-based fraud detection, ML-enhanced data generation, and analytics visualization.

**Core workflow**: Generate synthetic fraud data → Load into Neptune graph DB → Apply ML enhancements → Analyze with QuickSight dashboards

## Development Environment

**Language**: Python 3.x

**Installation**:
```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- AWS SDK: `boto3` for AWS service integration
- Graph: `gremlinpython`, `graph-notebook`, `networkx` for Neptune graph operations
- ML: `sklearn` (ai_learning_enhancer.py), `sagemaker` (sagemaker_fraud_enhancer.py)
- Data: `pandas`, `numpy`, `faker` for data generation
- Notebooks: `jupyter-client`, `ipywidgets` for interactive workflows

## Architecture

### Data Generation Pipeline

1. **Rule-Based Generation with Realistic Normal Transactions** (`enhanced_fraud_generator.py`):
   - Reads YAML configuration from `config/enhanced_fraud_rules.yaml`
   - Generates institutions, accounts, and transactions with fraud patterns
   - **NEW: Realistic Normal Transaction Generation** (enabled by default):
     - Uses account behavior profiles (`src/account_profiles.py`) with frequency tiers
     - Builds relationship networks between accounts (80% of transactions use existing relationships)
     - Implements log-normal amount distributions based on account type and relationship
     - Generates temporal patterns: day-of-week preferences, account-specific hours, intra-day clustering
     - Creates transaction sequences: recurring payments, transaction chains, batch payments
   - Supports 9 fraud types: money_laundering_ring, smurfing_network, shell_company_web, trade_based_laundering, cryptocurrency_mixing, insider_fraud, synthetic_identity, unusual_amount, off_hours
   - Produces CSV outputs in `enhanced_output/` directory

2. **AI Enhancement Layer**:
   - **Local AI** (`ai_learning_enhancer.py`): Uses sklearn IsolationForest to learn from existing data and generate additional transactions with AI confidence scores
   - **SageMaker AI** (`sagemaker_fraud_enhancer.py`): Uses SageMaker RandomCutForest and XGBoost for more sophisticated pattern learning and anomaly detection

3. **Neptune Loading**:
   - **Interactive** (`enhanced_neptune_loader.py`): Direct Gremlin API loading for small datasets
   - **Bulk** (`neptune_bulk_loader.py`): S3-based Neptune bulk loader for large datasets (100K+ transactions)
   - Supports incremental loading (RESUME mode) to add AI-enhanced data to existing graphs

### Graph Schema

**Vertices**:
- `Institution`: Financial institutions (banks, credit unions, fintechs)
- `Account`: Customer accounts with risk indicators (is_shell, is_synthetic)
- `FraudPattern`: Aggregate fraud pattern nodes

**Edges**:
- `HOLDS_ACCOUNT`: Institution → Account
- `PAYMENT`: Account → Account (transactions with fraud metadata)
- `MATCHES_PATTERN`: Transaction → FraudPattern

### Realistic Normal Transaction Generation Architecture (NEW)

The system now generates highly realistic normal transactions that simulate real-world banking behavior patterns. This replaces the previous simplistic random transaction generation.

**Key Components**:

1. **Account Behavior Profiles** (`src/account_profiles.py`):
   - **Frequency Tiers**: dormant, low, medium, high, very_high (controls transaction volume)
   - **Account-Specific Patterns**: Each account has preferred hours, preferred days, amount ranges
   - **Account Type Behaviors**: Checking, savings, business, and trust accounts have distinct patterns
   - **Relationship Preferences**: Intra-institution vs cross-institution transaction preference

2. **Relationship Networks**:
   - **Relationship Types**: recurring_payment, bill_payment, peer_transfer, business_payment, salary_deposit, one_time
   - **Relationship Strength**: 0-1 value determining transaction frequency between account pairs
   - **Network Building**: Automatically creates realistic networks based on account types and institution groupings
   - 80% of transactions use existing relationships (configurable)

3. **Realistic Amount Distributions**:
   - **Log-Normal Distribution**: Replaces uniform distribution for more realistic amounts
   - **Account Type Variance**: Business accounts have larger amounts than personal accounts
   - **Relationship-Based Amounts**: Bill payments rounded to $50, salaries rounded to $100
   - **Frequency-Amount Correlation**: High-frequency accounts have smaller average amounts

4. **Temporal Patterns**:
   - **Day-of-Week Effects**: Weekday vs weekend activity patterns
   - **Account-Specific Hours**: Some accounts active mornings, others evenings
   - **Intra-Day Clustering**: Transactions cluster around :00, :15, :30, :45 minutes
   - **Business Hours Modeling**: Business accounts restricted to weekday business hours

5. **Transaction Sequences**:
   - **Recurring Transactions**: Monthly, weekly, or bi-weekly scheduled payments
   - **Transaction Chains**: 15% of transactions trigger follow-up transactions (A→B→C→D)
   - **Batch Payments**: High-frequency accounts generate payment batches (3-10 payments within 5-60 minutes)

**Configuration Control**:
- Enable/disable realistic generation: `normal_transactions.use_realistic_generation` (default: true)
- Adjust relationship probability: `normal_transactions.relationship_based_probability` (default: 0.8)
- Configure temporal patterns: `temporal_patterns` section in YAML
- Fine-tune behavior profiles: `account_behavior` section in YAML

**Backward Compatibility**: Setting `use_realistic_generation: false` reverts to simple random generation.

### Configuration System

**Primary config**: `config/enhanced_fraud_rules.yaml`
- Controls network structure (institutions, accounts, transactions counts)
- Defines fraud type distributions and characteristics
- Sets risk scoring thresholds and multipliers
- Configurable fraud rate (default: 3% of transactions)
- **NEW: Account behavior profiles** (`account_behavior` section)
- **NEW: Relationship network settings** (`relationship_network` section)
- **NEW: Temporal patterns configuration** (`temporal_patterns` section)

**Neptune config**: `config/neptune_config.json`
- Contains Neptune cluster endpoint and S3 bucket configuration

**Domain rules**: `config/domain_rules.yaml`, `config/financial_network_rules.yaml`
- Additional rules for specific fraud patterns (not actively used in main workflows)

## Common Workflows

### Running Notebooks (Sequential Order)

1. **Enhanced_Fraud_Bulk_Load_Workflow.ipynb**: Generate and bulk load 100K transactions
   - Generates base dataset using `EnhancedFraudGenerator`
   - Prepares Neptune CSV format
   - Uploads to S3 and initiates bulk load
   - Typical runtime: 5-10 minutes

2. **AI_Enhanced_Generation.ipynb**: Add 30K AI-learned transactions
   - Trains AI models on existing 100K transactions
   - Generates additional transactions with pattern learning
   - Incrementally loads to Neptune
   - Adds ai_confidence and pattern_similarity metrics

3. **Simple_SageMaker_AI_Enhanced_Generation.ipynb**: SageMaker-based enhancement
   - Alternative to local AI enhancement
   - Uses SageMaker training jobs for RandomCutForest and XGBoost
   - Better for larger datasets and production scenarios

4. **Fraud_Detection_Analytics.ipynb**: Analyze fraud patterns
   - Gremlin queries for fraud detection
   - Pattern analysis across fraud types
   - Risk scoring validation

5. **QuickSight_Programmatic_Dashboard.ipynb**: Create visualization dashboards
   - Programmatically creates QuickSight data sources, datasets, and dashboards
   - Uses `create_quicksight_dashboard.py` and `prepare_quicksight_data.py`

### Direct Script Usage

**Generate fraud data**:
```python
from src.enhanced_fraud_generator import EnhancedFraudGenerator

generator = EnhancedFraudGenerator('config/enhanced_fraud_rules.yaml')
data = generator.generate_network()  # Returns dict with 'institutions', 'accounts', 'transactions' DataFrames
```

**Load to Neptune (bulk)**:
```python
from src.neptune_bulk_loader import NeptuneBulkLoader

loader = NeptuneBulkLoader(
    neptune_endpoint='your-cluster.neptune.amazonaws.com',
    s3_bucket='your-bucket',
    neptune_role_arn='arn:aws:iam::123456789:role/NeptuneS3Role'
)
loader.bulk_load_enhanced_fraud_data('enhanced_output')
```

**AI enhancement**:
```python
from src.ai_learning_enhancer import AILearningEnhancer

enhancer = AILearningEnhancer()
enhancer.load_existing_data('enhanced_output')
enhancer.train_ai_models()
ai_transactions = enhancer.generate_ai_enhanced_transactions(30000)
```

## AWS Integration Requirements

**Neptune**: Graph database cluster with VPC endpoint access
**S3**: Bucket for Neptune bulk loading (naming: `{account_id}-neptune-bulk-load`)
**IAM**: Neptune requires S3 access role (typically `NeptuneS3AccessRole`)
**SageMaker**: (Optional) For SageMaker-based AI enhancement
**QuickSight**: (Optional) For dashboard visualization

## Output Data Structure

### Generated CSV Files

**institutions.csv**: institution_id, name, type, channels, routing_number, swift_code, fedline_tier
**accounts.csv**: account_id, institution_id, account_type, balance, is_shell, is_synthetic
**transactions.csv**: Full payment system schema with 55 fields (51 required + 4 fraud metadata)

### Transaction Schema Compliance

Generated transactions comply with the payment system schema defined in `config/schema.txt`. All 51 required fields are included:

**Transaction Identifiers**:
- payment_rail, transaction_id_1, transaction_id_2, reference_id, previous_message_id

**Originating/Sending Details** (ODFI - Originating Depository Financial Institution):
- odfi, odfi_name, sending_fedline_tier, sending_net_source
- instructing_fi_id_code, instructing_fi_id, instructing_fi_name
- ofi_id_code, ofi_id, ofi_name
- originating_account_type, originating_account_identifier, originating_account_description

**Receiving Details** (RDFI - Receiving Depository Financial Institution):
- rdfi, rdfi_name, receiving_fedline_tier, receiving_net_source
- intermediary_fi_id_code, intermediary_fi_id, intermediary_fi_name
- rfi_id_code, rfi_id, rfi_name
- receiving_account_type, receiving_account_identifier, receiving_account_description

**Payment Type Details**:
- pay_type, pay_type_subtype_1, pay_type_subtype_2, pay_code, pay_code_type
- payment_speed_category (STANDARD for ACH, EXPEDITED for WIRE, INSTANT for RTP)
- return_/_error_reason (empty for successful transactions)

**Transaction Metadata**:
- originator_type (INDIVIDUAL, CORPORATE, FIDUCIARY)
- transaction_purpose (BILL_PAYMENT, PAYROLL, PERSON_TO_PERSON, etc.)
- value_type, recurring_payment

**Dates and Timestamps**:
- transaction_create_date, transaction_processed_date
- odfi_transaction_settlement_date, rdfi_transaction_settlement_date

**Amounts and Directions**:
- dollar_value, security_par_value
- dollar_direction, security_direction
- addenda_count

**Additional Fraud Metadata** (4 extra fields):
- is_fraud, fraud_type, risk_score, relationship_type

**Validation**: Run `python test_schema.py` to validate schema compliance

### Neptune Bulk Load Format

Files in `bulk_load_data/` directory use Neptune CSV conventions:
- Vertices: Use `~id` and `~label` columns with typed properties (`:String`, `:Double`, `:Boolean`, `:Int`)
- Edges: Use `~id`, `~from`, `~to`, `~label` columns

## Key Implementation Details

### Realistic Normal Transaction Generation (NEW)

The normal transaction generation has been completely redesigned to produce realistic patterns:

**Phase 1: Account Profile Generation**
- Each account is assigned a behavior profile with:
  - Frequency tier (dormant → very_high) based on account type distribution
  - Expected transactions per month (0.2-35 range)
  - Preferred activity hours (morning/lunch/evening/all-day patterns)
  - Preferred days (weekdays vs weekends, business accounts only M-F)
  - Amount mean and standard deviation based on frequency tier
  - Number of relationships to establish (0-20)

**Phase 2: Relationship Network Building**
- Creates directed graph of account relationships
- Relationship types assigned based on account type:
  - Business accounts: 60% business_payment, 30% recurring_payment, 10% bill_payment
  - Savings accounts: 70% recurring_payment, 30% peer_transfer
  - Personal accounts: 40% bill_payment, 30% peer_transfer, 20% recurring_payment, 10% business_payment
- Respects intra-institution preferences (typically 65%)
- Each relationship has strength value (0-1) affecting selection probability

**Phase 3: Transaction Generation**
1. **Recurring Transactions** (if enabled):
   - Relationships of type recurring_payment, bill_payment, or salary_deposit generate scheduled transactions
   - Monthly (30-day), weekly (7-day), or bi-weekly (14-day) intervals
   - Small variance (±2 days) for realism

2. **Profile-Based Allocation**:
   - Total transaction count distributed across accounts proportional to their frequency profiles
   - High-frequency accounts generate more transactions than dormant accounts
   - Ensures realistic volume distribution

3. **Individual Transaction Creation**:
   - Sender selection based on allocation
   - Receiver selection: 80% from relationship network, 20% random
   - Amount: Log-normal distribution adjusted for relationship type
   - Timing: Account-specific hour preferences + day-of-week patterns
   - Minute clustering around :00, :15, :30, :45 for realism

4. **Batch Payments** (if enabled):
   - 8% probability for high/very-high frequency accounts
   - Generates 3-10 payments within 5-60 minute window
   - All from same sender, realistic for business operations

5. **Transaction Chains** (if enabled):
   - 15% of transactions trigger follow-up chain
   - Chain length: 2-4 transactions
   - Each transaction uses receiver as next sender (A→B, B→C, C→D)
   - Time window: 1-48 hours between chain steps

**Amount Generation Details**:
```python
# Base amount from log-normal distribution
amount = lognormal(mean=profile.amount_mean, std=profile.amount_std)

# Adjustments by relationship type:
# - Bill payments: Round to nearest $50 (e.g., $150, $200, $250)
# - Salary deposits: 3x base, min $1,500, round to $100
# - Peer transfers: 0.6x base (smaller amounts)
# - Business payments: 1.5x base (larger amounts)
```

**Timing Generation Details**:
- Day selection: Random within 90-day period, then 70% reroll if not in preferred days
- Hour selection: 80% from profile's preferred hours, 20% random 6am-10pm
- Minute selection: Cluster around :00/:15/:30/:45 with ±5 minute variance

**Transaction Output Fields**:
- Standard fields: transaction_id, sender_account, receiver_account, amount, payment_type, timestamp, etc.
- **NEW**: `relationship_type` field indicates the type of relationship used (or 'one_time')

### Fraud Pattern Generation

Each fraud type has specific characteristics defined in YAML:
- **money_laundering_ring**: Circular transfers between 3-8 accounts
- **smurfing_network**: Multiple transactions just under $10K threshold
- **shell_company_web**: Complex layering through shell accounts (2-6 layers)
- **trade_based_laundering**: High-value transactions with price manipulation (1.5-5x)
- **synthetic_identity**: Uses accounts flagged as `is_synthetic=True`

The generator creates fraud rings/networks first, then generates individual transactions based on ring membership.

### Risk Scoring Algorithm

Risk scores (0-100) calculated from multiple factors:
- **Amount risk** (max 30 points): Thresholds at $100K, $50K, $10K, and unusually small amounts
- **Time risk** (max 20 points): Off-hours (10PM-6AM), extended hours (6-7AM, 6-9PM)
- **Account risk** (max 25 points): Shell or synthetic accounts as sender/receiver
- **Cross-institution risk** (max 15 points): Different sender/receiver institutions
- **Fraud type multipliers** (1.3-2.0x): Applied based on fraud severity

### Bulk Loading Performance

Neptune bulk loader uses S3-based ingestion for optimal performance:
- Prepares separate CSV files for vertex types and edge types
- Uploads to S3 with prefix structure
- Creates Neptune bulk load job via REST API
- Polls job status until completion
- Supports OVERWRITE (replace all) or RESUME (incremental) modes

For 100K transactions: ~2-3 minutes total (preparation + upload + loading)

### AI Enhancement Approach

**Pattern Learning**:
- Learns fraud type distribution from existing data
- Tracks timing patterns per fraud type (hour-of-day frequency)
- Builds amount prediction models per fraud type

**Anomaly Detection**:
- Trains IsolationForest on transaction features
- Uses anomaly scores to enhance risk scores
- Provides confidence metrics for generated data

**Quality Metrics**:
- `ai_confidence`: Model confidence in fraud classification (0-1)
- `pattern_similarity`: Similarity to historical patterns (0-1)
- `generation_method`: "AI_Enhanced" tag for tracking

## File Organization

```
/
├── config/                       # Configuration files
│   ├── enhanced_fraud_rules.yaml # Main fraud generation config
│   ├── neptune_config.json       # Neptune connection settings
│   ├── odfi_rdfi_matrix.csv      # NEW: Institutional relationship matrix (optional)
│   ├── schema.txt                # Payment system schema definition
│   └── *.yaml                    # Additional domain rules
├── queries/                      # NEW: SQL queries for data analysis
│   └── generate_odfi_rdfi_matrix.sql  # Generate institutional relationship matrix
├── src/                          # Python modules
│   ├── enhanced_fraud_generator.py       # Rule-based + realistic generation
│   ├── account_profiles.py               # NEW: Account behavior profiles & relationship networks
│   ├── institutional_relationships.py    # NEW: ODFI-RDFI matrix manager
│   ├── schema_mapper.py                  # NEW: Payment system schema compliance mapper
│   ├── ai_learning_enhancer.py           # Local AI enhancement
│   ├── sagemaker_fraud_enhancer.py       # SageMaker AI enhancement
│   ├── neptune_bulk_loader.py            # S3 bulk loading
│   ├── enhanced_neptune_loader.py        # Direct Gremlin loading
│   ├── create_quicksight_dashboard.py    # Dashboard creation
│   ├── prepare_quicksight_data.py        # QuickSight data prep
│   └── xgboost_fraud_script.py           # SageMaker training script
├── *.ipynb                       # Jupyter notebooks (main workflows)
├── requirements.txt              # Python dependencies
└── enhanced_output/              # Generated data (CSV files)
```

### Institutional Relationship Matrix (NEW)

The system can use real-world transaction volume patterns between financial institutions to generate realistic ODFI-RDFI (Originating→Receiving institution) distributions.

**How It Works**:
1. **Matrix Generation**: Run `queries/generate_odfi_rdfi_matrix.sql` against your existing transaction database to create a CSV matrix of institution-to-institution transaction volumes
2. **Automatic Loading**: Place the matrix at `config/odfi_rdfi_matrix.csv` and it will be automatically loaded during generation
3. **Pattern-Based Selection**: When generating transactions, the system selects receiving institutions based on observed volume patterns from the matrix
4. **Fallback**: If no matrix is available, a sample matrix is auto-generated for testing

**Matrix Fields**:
```
odfi, rdfi, transaction_count, total_dollar_value, avg_dollar_value,
ach_percentage, wire_percentage, rtp_percentage, recurring_percentage,
avg_settlement_days, relationship_strength
```

**Configuration**:
```yaml
institutional_relationships:
  use_matrix: true  # Enable matrix-based selection
  matrix_path: "config/odfi_rdfi_matrix.csv"
  create_sample_if_missing: true  # Auto-generate for testing
  matrix_based_probability: 0.90  # 90% use matrix, 10% random
```

**Expected Coverage**:
- 40-60% of generated transactions will use exact ODFI-RDFI pairs from the matrix
- Remaining transactions use same-institution transfers or account-level relationships
- This mimics real-world behavior where not all transfers follow bulk patterns

**Testing**: Run `python test_institutional_relationships.py` to validate matrix usage

### Payment System Schema Mapping (NEW)

After transactions are generated, they are automatically mapped to the full payment system schema defined in `config/schema.txt`. The `PaymentSchemaMapper` class (`src/schema_mapper.py`) handles this transformation:

**Institution Code Generation**:
- Routing numbers: 9-digit codes based on institution type (e.g., major banks start with 011/021/026)
- SWIFT codes: 8-character codes generated from institution names
- FedLine tiers: TIER_1, TIER_2, or TIER_3 randomly assigned

**Settlement Date Calculation**:
- **RTP**: Instant settlement (same timestamp for all dates)
- **WIRE**: Same-day settlement with 2-hour processing delay
- **ACH**: 1-2 business day settlement, accounts for weekends

**Payment Type Mapping**:
```
ACH:  pay_code=CCD,  payment_speed=STANDARD
WIRE: pay_code=CTR,  payment_speed=EXPEDITED
RTP:  pay_code=P2P,  payment_speed=INSTANT
```

**Account Type Mapping**:
```
checking → DDA (Demand Deposit Account)
savings  → SAV (Savings)
business → DDA with Commercial subtype
trust    → TRU (Trust)
```

**Transaction Purpose Inference**:
- Derives purpose from relationship_type:
  - bill_payment → BILL_PAYMENT
  - salary_deposit → PAYROLL
  - peer_transfer → PERSON_TO_PERSON
  - business_payment → TRADE_SETTLEMENT

**Originator Type**:
- Business accounts → CORPORATE
- Trust accounts → FIDUCIARY
- Other accounts → INDIVIDUAL

The schema mapping is applied automatically in `EnhancedFraudGenerator.generate_network()` after all transactions are generated. To validate schema compliance, run `python test_schema.py`.

## Important Notes

- **Data Scale**: Default generates 100K transactions. Adjust `total_transactions` in YAML for different scales.
- **Fraud Rate**: Default 3% fraud rate. Configurable via `fraud_patterns.overall_fraud_rate` in YAML.
- **Neptune Endpoint**: Must be updated in notebooks (default shows 'UPDATE-ME' placeholder).
- **S3 Bucket Naming**: Auto-generated as `{account_id}-neptune-bulk-load`. Must have Neptune IAM role access.
- **Workshop Context**: This is designed for AWS workshops/demos, not production fraud detection.
