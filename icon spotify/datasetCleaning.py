def cleanAndRenameColumnsOfDataset(dataSet, differentialColumn):
    #vecchio codice
    dataSet.rename(columns={
        'Col1': 'pelvic incidence',
        'Col2': 'pelvic tilt',
        'Col3': 'lumbar lordosis angle',
        'Col4': 'sacral slope',
        'Col5': 'pelvic radius',
        'Col6': 'degree spondylolisthesis',
        'Col7': 'pelvic slope',
        'Col8': 'direct tilt',
        'Col9': 'thoracic slope',
        'Col10': 'cervical tilt',
        'Col11': 'sacrum angle',
        'Col12': 'scoliosis slope',
        'Class_att': "spine_state"
    },
        inplace=True, errors='raise')

    dataSet[differentialColumn] = dataSet[differentialColumn].replace(
        "Abnormal", 0)
    dataSet[differentialColumn] = dataSet[differentialColumn].replace(
        "Normal", 1)

    dataSet = dataSet.loc[:, ~dataSet.columns.str.contains(
        '^Unnamed')]  # Removal of empty column

    dataSet = dataSet.reset_index(drop=True)

    return dataSet


def dataOverview(dataSet):
    print("\nDisplay (partial) of the dataframe:\n", dataSet.head())
    print("\nNumber of elements: ", len(dataSet.index) - 1)
    print("\nInfo dataset:\n", dataSet.describe())
