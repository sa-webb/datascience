import csv
import functools
import os.path

import numpy
from sqlalchemy import create_engine
import pandas
from sklearn_pandas import DataFrameMapper
import sklearn.preprocessing
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn import metrics


DIR = '/home/user/workspace/machine-learning'

TRAIN_DATA_FILE_PATH = os.path.join(DIR, 'data', 'adult.data')
TEST_DATA_FILE_PATH = os.path.join(DIR, 'data', 'adult.test')

TRAIN_DB_FILE_PATH = os.path.join(DIR, 'db', 'data.sqlite')
TEST_DB_FILE_PATH = os.path.join(DIR, 'db', 'test.sqlite')

train_engine = create_engine(f'sqlite:///{TRAIN_DB_FILE_PATH}')
test_engine = create_engine(f'sqlite:///{TEST_DB_FILE_PATH}')

INT = 'INTEGER'
STR = 'VARCHAR'

FIELDS = (
    ('age', INT),
    ('workclass', STR),
    ('fnlwgt', INT),
    ('education', STR),
    ('education_num', INT),
    ('marital_status', STR),
    ('occupation', STR),
    ('relationship', STR),
    ('race', STR),
    ('sex', STR),
    ('capital_gain', INT),
    ('capital_loss', INT),
    ('hours_per_week', INT),
    ('native_country', STR),
    ('classification', STR)
)


def create_schema(connection):
    fields_sql = ', '.join(
        f'{field_name} {field_type}' for (field_name, field_type) in FIELDS
    )
    connection.execute(
        f'CREATE TABLE adult (id INTEGER PRIMARY KEY, {fields_sql})'
    )


def read_data(data_file_path):
    with open(data_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        for row in reader:
            if len(row) != 15:
                continue  # Skip empty rows, skip test file header
            classification = row[-1]
            if classification.endswith('.'):
                # Test file has dots ('.') at the end of lines, strip them out.
                row[-1] = classification[:-1]
            yield row


def insert_row(row, connection):
    fields = ', '.join(field_name for (field_name, _) in FIELDS)
    placeholders = ', '.join(['?'] * len(FIELDS))
    connection.execute(
        f'INSERT INTO adult ({fields}) VALUES ({placeholders})', row
    )


def import_data(data, connection):
    create_schema(connection)
    with connection.begin():
        for row in data:
            insert_row(row, connection)


def gather_data():
    return read_data(TRAIN_DATA_FILE_PATH), read_data(TEST_DATA_FILE_PATH)


def store_data(train_data, test_data):
    with train_engine.connect() as conn:
        import_data(train_data, conn)
    with test_engine.connect() as conn:
        import_data(test_data, conn)


def load_data(train_engine, test_engine):
    with train_engine.connect() as conn:
        with conn.begin():
            train_data_frame = pandas.read_sql_table(
                'adult', conn, index_col='id'
            )

    with test_engine.connect() as conn:
        with conn.begin():
            test_data_frame = pandas.read_sql_table(
                'adult', conn, index_col='id'
            )

    return train_data_frame, test_data_frame


def get_mapper():
    def numpy_map(callback):
        @functools.wraps(callback)
        def numpy_map_wrapper(X):
            return numpy.array([callback(x) for x in X])
        return numpy_map_wrapper

    @numpy_map
    def native_country_generalize(x):
        return 'US' if x == 'United-States' else 'Other'

    @numpy_map
    def workclass_generalize(x):
        if x in ['Self-emp-not-inc', 'Self-emp-inc']:
            return 'Self-emp'
        elif x in ['Local-gov', 'State-gov', 'Federal-gov']:
            return 'Gov'
        elif x in ['Without-pay', 'Never-worked', '?']:
            return 'None'
        else:
            return x

    @numpy_map
    def education_generalize(x):
        if x in ['Assoc-voc', 'Assoc-acdm']:
            return 'Assoc'
        elif x in [
            '11th', '10th', '7th-8th', '9th', '12th', '5th-6th',
            '1st-4th', 'Preschool'
        ]:
            return 'Low'
        else:
            return x

    return DataFrameMapper([
        (['age'], sklearn.preprocessing.StandardScaler()),
        ('workclass', [
            sklearn.preprocessing.FunctionTransformer(
                workclass_generalize, validate=False
            ),
            sklearn.preprocessing.LabelBinarizer()
        ]),
        # ('fnlwgt', None),
        ('education', [
            sklearn.preprocessing.FunctionTransformer(
                education_generalize, validate=False
            ),
            sklearn.preprocessing.LabelBinarizer()
        ]),
        (['education_num'], sklearn.preprocessing.StandardScaler()),
        ('marital_status', sklearn.preprocessing.LabelBinarizer()),
        ('occupation', sklearn.preprocessing.LabelBinarizer()),
        ('relationship', sklearn.preprocessing.LabelBinarizer()),
        ('race', sklearn.preprocessing.LabelBinarizer()),
        ('sex', sklearn.preprocessing.LabelBinarizer()),
        (['capital_gain'], sklearn.preprocessing.StandardScaler()),
        (['capital_loss'], sklearn.preprocessing.StandardScaler()),
        (['hours_per_week'], sklearn.preprocessing.StandardScaler()),
        ('native_country', [
            sklearn.preprocessing.FunctionTransformer(
                native_country_generalize, validate=False
            ),
            sklearn.preprocessing.LabelBinarizer()
        ]),
    ])


classification_map = {
    '<=50K': True,
    '>50K': False
}


def train(train_data_frame, mapper):
    train_X = train_data_frame[train_data_frame.columns.drop('classification')]
    train_y = train_data_frame['classification'].map(classification_map)

    NUMBER_OF_LAYERS = 1
    NEURONS_PER_LAYER = 20

    classifier = MLPClassifier(
        hidden_layer_sizes=(NEURONS_PER_LAYER, ) * NUMBER_OF_LAYERS,
        alpha=0.01,
        random_state=1
    )

    pipeline = Pipeline([
        ('mapper', mapper),
        ('classifier', classifier)
    ])

    model = pipeline.fit(X=train_X, y=train_y)
    return model


def predict(model, test_data_frame):
    test_X = test_data_frame[test_data_frame.columns.drop('classification')]

    predictions = model.predict(X=test_X)
    return predictions


def assess(test_data_frame, predictions):
    test_y = test_data_frame['classification'].map(classification_map)

    accuracy_score = metrics.accuracy_score(test_y, predictions)
    return accuracy_score


def main():
    train_data, test_data = gather_data()
    store_data(train_data, test_data)
    train_data_frame, test_data_frame = load_data(train_engine, test_engine)
    mapper = get_mapper()
    model = train(train_data_frame, mapper)
    predictions = predict(model, test_data_frame)
    score = assess(test_data_frame, predictions)
    print('Accuracy score', score)


if __name__ == '__main__':
    main()