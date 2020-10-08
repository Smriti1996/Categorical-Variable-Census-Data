INPUT_FILE = '../input/adult-training.csv'

TRAINING_FILE = '../input/adult-training_folds.csv'

MODEL_OUTPUT = '../models'

COLUMN_NAMES = [
    'age', 
    'workclass',
    'fnlwgt',
    'education',
    'education_num',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country',
    'income_bracket'
    ]

# list of numerical columns to be removed
NUM_COLS = [
    'age',
    'fnlwgt',
    'capital_gain',
    'capital_loss',
    'hours_per_week'
]