import os
import pickle
import argparse
import xml.etree.ElementTree as ET
from InkmlToImage import InkmlToImage
from tqdm import tqdm


def read_args():
    version_choices = ['2011', '2012', '2013', '2014']
    category_names = ["all", "digits", "operators", "lowercase_letters",
                      "uppercase_letters", "greek", "miscellaneous", "symbols"]
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--data_path", required=True,
                    help="Path to find the available datasets.")
    ap.add_argument("-o", "--out_path", required=True,
                    help="Path to save the outputs.")
    ap.add_argument('-b', '--box_size', required=True, type=int,
                    help="Specify the length of a square box side.")
    ap.add_argument("-v", "--version", required=True, type=str,
                    help="Choose dataset version from list of available.", choices=version_choices, nargs="+")
    ap.add_argument('-c', '--category', required=True,
                    help="Specify what label category needs to be extracted", choices=category_names, nargs='+')
    ap.add_argument("--save_labels", action='store_true',
                    help="Whether to save the extracted labels.")
    args = vars(ap.parse_args())
    return args


def load_required_classes():
    # Load categories from `categories.txt` file
    categories = [{'name': cat.split(':')[0], 'classes': cat.split(
        ':')[1].split()} for cat in list(open(os.path.join(args["data_path"], 'categories.txt'), 'r'))]
    category_names = [cat['name'] for cat in categories]

    # Get classes that have to be extracted (based on categories selected by user)
    classes_to_extract = []
    for cat_name in args['category']:
        cat_idx = category_names.index(cat_name)
        classes_to_extract += categories[cat_idx]['classes']
    return classes_to_extract


def list_all_files(dir):
    all_files = []
    for path, dirs, files in os.walk(dir):
        all_files += [os.path.join(path, file) for file in files]
    return all_files


def load_inkml_files():
    train_inkml_files, test_inkml_files = [], []
    for version in args["version"]:
        folders = list(filter(lambda dir: version in dir.lower() and (
            "train" in dir.lower() or "test" in dir.lower() or "training" in dir.lower()), os.listdir(args["data_path"])))
        for folder in folders:
            if "train" in folder.lower() or "training" in folder.lower():
                files = list_all_files(os.path.join(args["data_path"], folder))
                train_inkml_files.extend(files)
            if "test" in folder.lower():
                files = list_all_files(os.path.join(args["data_path"], folder))
                test_inkml_files.extend(files)
    return train_inkml_files, test_inkml_files


def extract_trace_grps(inkml_file_abs_path, xmlns='{http://www.w3.org/2003/InkML}'):

    traces_data = []

    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()
    # doc_namespace = "{http://www.w3.org/2003/InkML}"
    doc_namespace = xmlns

    'Stores traces_all with their corresponding id'
    traces_all = [{'id': trace_tag.get('id'),
                   'coords': [[round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000)
                               for axis_coord in coord[1:].split(' ')] if coord.startswith(' ')
                              else [round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000)
                                    for axis_coord in coord.split(' ')]
                              for coord in (trace_tag.text).replace('\n', '').split(',')]}
                  for trace_tag in root.findall(doc_namespace + 'trace')]

    'Sort traces_all list by id to make searching for references faster'
    traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))

    'Always 1st traceGroup is a redundant wrapper'
    traceGroupWrapper = root.find(doc_namespace + 'traceGroup')

    if traceGroupWrapper is not None:
        for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):

            label = traceGroup.find(doc_namespace + 'annotation').text

            'traces of the current traceGroup'
            traces_curr = []
            for traceView in traceGroup.findall(doc_namespace + 'traceView'):

                'Id reference to specific trace tag corresponding to currently considered label'
                traceDataRef = int(traceView.get('traceDataRef'))

                'Each trace is represented by a list of coordinates to connect'
                single_trace = traces_all[traceDataRef]['coords']
                traces_curr.append(single_trace)

            traces_data.append({'label': label, 'traces': traces_curr})

    else:
        'Consider Validation data that has no labels'
        [traces_data.append({'traces': [trace['coords']]})
         for trace in traces_all]

    return traces_data


def apply_data_formatting(inkml_files, req_classes=None):
    data = []
    damaged_classes = []
    classes = set()
    convertor = InkmlToImage()
    for file in tqdm(inkml_files):
        try:
            trace_groups = extract_trace_grps(file)
            patterns_enc, classes_rejected, classes = convertor.convert_to_img(
                trace_groups, req_classes=req_classes, box_size=args["box_size"])
            if classes_rejected:
                damaged_classes.append(classes_rejected)
            if patterns_enc:
                data.append(patterns_enc)
            if classes:
                classes.update(classes)
        except Exception as e:
            print(e)
            continue
    return data, damaged_classes, classes


def get_batches(items, batch_size=300):
    for i in range(0, len(items), batch_size):
        yield items[i:i+batch_size]


def write_to_disk(data, index, train=True):
    if train:
        train_out_dir = os.path.join(args["out_path"], 'train')
        if not os.path.exists(train_out_dir):
            os.makedirs(train_out_dir)
        with open(os.path.join(train_out_dir, 'train_{}.pickle'.format(index)), 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('Training batch_no {} has been successfully dumped into'.format(
                index), f.name)

    else:
        test_out_dir = os.path.join(args["out_path"], 'test')
        if not os.path.exists(test_out_dir):
            os.makedirs(test_out_dir)
        with open(os.path.join(test_out_dir, 'test_{}.pickle'.format(index)), 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('Testing batch_no {} has been successfully dumped into'.format(
                index), f.name)


def process_files(files, req_classes=None, train=True):
    all_classes = set()
    for i, batch in enumerate(get_batches(files)):
        print("Batch no: ", i+1)
        pbatch_files, damaged_files, classes = apply_data_formatting(
            batch, req_classes=req_classes)
        all_classes.update(classes)
        print('Processed Batch set size:', len(pbatch_files))
        print('How many rejected trace groups:', len(damaged_files))

        write_to_disk(pbatch_files, i+1, train=train)
    return all_classes


if __name__ == "__main__":
    args = read_args()
    classes_to_extract = load_required_classes()
    train_inkml_files, test_inkml_files = load_inkml_files()
    print('Number of training INKML files:', len(train_inkml_files))
    print('Number of testing INKML files:', len(test_inkml_files))

    train_classes = process_files(
        train_inkml_files, classes_to_extract, train=True)
    test_classes = process_files(
        test_inkml_files, classes_to_extract, train=False)
    print("Done Processing")

    if args["save_labels"]:
        classes = train_classes.union(test_classes)
        # Save all labels in 'classes.txt' file
        with open(os.path.join(args["data_path"], 'classes.txt'), 'w') as f:
            f.write("\n".join(classes))
            print('All classes that were extracted are listed in: {} file, in the path: {}'.format(
                f.name, args["data_path"]))
