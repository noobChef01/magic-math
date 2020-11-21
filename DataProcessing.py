import os
import pickle
import argparse
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from tqdm import tqdm
from skimage.draw import line
from skimage.morphology import thin
import matplotlib.pyplot as plt
from io import StringIO


version_choices = ['2011', '2012', '2013', '2014']
# Load categories from `categories.txt` file
# categories = [{'name': cat.split(':')[0], 'classes': cat.split(
#     ':')[1].split()} for cat in list(open('categories.txt', 'r'))]
# category_names = [cat['name'] for cat in categories]

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--data_path", required=True,
                help="Path to find the available datasets.")
ap.add_argument('-b', '--box_size', required=True, type=int,
                help="Specify the length of a square box side.")
ap.add_argument("-v", "--version", required=True, type=str,
                help="Choose dataset version from list of available.", choices=version_choices, nargs="+")
# ap.add_argument('-c', '--category', required=True,
#                 help="Specify what label category needs to be extracted", choices=category_names, nargs='+')
args = vars(ap.parse_args())

# # Get classes that have to be extracted (based on categories selected by user)
# classes_to_extract = []
# for cat_name in args.get('category'):
#     cat_idx = category_names.index(cat_name)
#     classes_to_extract += categories[cat_idx]['classes']


def get_files_within_subdir(files, path):
    result = []
    for item in files:
        if item.endswith(".inkml"):
            result.append(item)
        if os.path.isdir(item):
            result.append([os.path.join(item, file)
                           for file in os.listdir(os.path.join(path, item))])
    return result


def extract_inkml_files():
    train_inkml_files, test_inkml_files = [], []
    for version in args["version"]:
        folders = list(filter(lambda dir: version in dir.lower() and (
            "train" in dir.lower() or "test" in dir.lower() or "training" in dir.lower()), os.listdir(args["data_path"])))
        for folder in folders:
            if "train" in folder.lower() or "training" in folder.lower():
                files = os.listdir(os.path.join(args["data_path"], folder))
                path = os.path.join(args["data_path"], folder)
                files = get_files_within_subdir(files, path)
                train_inkml_files.extend(
                    [os.path.join(args["data_path"], folder, file) for file in files])
            if "test" in folder.lower():
                files = os.listdir(os.path.join(args["data_path"], folder))
                path = os.path.join(args["data_path"], folder)
                files = get_files_within_subdir(files, path)
                test_inkml_files.extend(
                    [os.path.join(args["data_path"], folder, file) for file in files])
    return train_inkml_files, test_inkml_files


def inkml2img(input_path, output_path=None, color='black'):
    traces = extract_trace_grps(input_path)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    plt.axes().spines['top'].set_visible(False)
    plt.axes().spines['right'].set_visible(False)
    plt.axes().spines['bottom'].set_visible(False)
    plt.axes().spines['left'].set_visible(False)
    for elem in traces:
        ls = elem['traces']
        for subls in ls:
            data = np.array(subls)
            x, y = zip(*data)
            plt.plot(x, y, linewidth=2, c=color)
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.gcf().clear()


def draw_pattern(trace_group, box_size):
    pattern_drawn = np.ones(shape=(box_size, box_size), dtype=np.float32)
    for trace in trace_group:
        # SINGLE POINT TO DRAW
        if len(trace) == 1:
            x_coord = trace[0][0]
            y_coord = trace[0][1]
            pattern_drawn[y_coord, x_coord] = 0.0
        else:
            # TRACE HAS MORE THAN 1 POINT
            # Iterate through list of traces endpoints
            for pt_idx in range(len(trace) - 1):
                print(pt_idx, trace[pt_idx])

                # Indices of pixels that belong to the line. May be used to directly index into an array
                pattern_drawn[line(r0=int(trace[pt_idx][1]), c0=int(trace[pt_idx][0]),
                                   r1=int(trace[pt_idx + 1][1]), c1=int(trace[pt_idx + 1][0]))] = 0
    return pattern_drawn


# def extract_trace_grps(inkml_file_abs_path):
#     trace_grps = []
#     tree = ET.parse(inkml_file_abs_path)
#     root = tree.getroot()
#     doc_namespace = "{http://www.w3.org/2003/InkML}"

#     # Find traceGroup wrapper - traceGroup wrapping important traceGroups
#     pinr
#     traceGrpWrapper = root.findall(doc_namespace + 'traceGroup')[0]
#     traceGroups = traceGrpWrapper.findall(doc_namespace + 'traceGroup')
#     for traceGrp in traceGroups:
#         latex_class = traceGrp.findall(doc_namespace + 'annotation')[0].text
#         traceViews = traceGrp.findall(doc_namespace + 'traceView')
#         # Get traceid of traces that refer to latex_class extracted above
#         id_traces = [traceView.get('traceDataRef') for traceView in traceViews]
#         # Construct pattern object
#         trace_grp = {'label': latex_class, 'traces': []}

#         # Find traces with referenced by latex_class
#         traces = [trace for trace in root.findall(
#             doc_namespace + 'trace') if trace.get('id') in id_traces]
#         # Extract trace coords
#         for idx, trace in enumerate(traces):
#             coords = []
#             for coord in trace.text.replace('\n', '').split(','):
#                 # Remove empty strings from coord list (e.g. ['', '-238', '-91'] -> [-238', '-91'])
#                 coord = list(filter(None, coord.split(' ')))
#                 # Unpack coordinates
#                 x, y = coord[:2]
#                 # print('{}, {}'.format(x, y))
#                 if not float(x).is_integer():
#                     # Count decimal places of x coordinate
#                     # d_places = len(x.split('.')[-1])
#                     # ! Get rid of decimal places (e.g. '13.5662' -> '135662')
#                     # x = float(x) * (10 ** len(x.split('.')[-1]) + 1)
#                     x = float(x) * 10000
#                 else:
#                     x = float(x)
#                 if not float(y).is_integer():
#                     # Count decimal places of y coordinate
#                     # d_places = len(y.split('.')[-1])
#                     # ! Get rid of decimal places (e.g. '13.5662' -> '135662')
#                     # y = float(y) * (10 ** len(y.split('.')[-1]) + 1)
#                     y = float(y) * 10000
#                 else:
#                     y = float(y)

#                 # Cast x & y coords to integer
#                 x, y = round(x), round(y)
#                 coords.append([x, y])
#             trace_grp['traces'].append(coords)
#         trace_grps.append(trace_grp)

#         # print('Pattern: {};'.format(pattern))
#     return trace_grps


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


def center_pattern(trace_group, max_x, max_y, box_size):
    x_margin = int((box_size - max_x) / 2)
    y_margin = int((box_size - max_y) / 2)
    return shift_trace_grp(trace_group, min_x=-x_margin, min_y=-y_margin)


def interpolate(trace_group, trace_grp_height, trace_grp_width, box_size):
    interpolated_trace_grp = []
    if trace_grp_height == 0:
        trace_grp_height += 1
    if trace_grp_width == 0:
        trace_grp_width += 1
    # KEEP original size ratio
    trace_grp_ratio = (trace_grp_width) / (trace_grp_height)
    scale_factor = 1.0
    # Set \"rescale coefficient\" magnitude
    if trace_grp_ratio < 1.0:
        scale_factor = (box_size / trace_grp_height)
    else:
        scale_factor = (box_size / trace_grp_width)
    for trace in trace_group:
        # coordintes convertion to int type necessary
        interpolated_trace = [
            [round(coord[0] * scale_factor), round(coord[1] * scale_factor)] for coord in trace]
        interpolated_trace_grp.append(interpolated_trace)
    return interpolated_trace_grp


def shift_trace_grp(trace_group, min_x, min_y):
    shifted_trace_grp = []
    for trace in trace_group:
        shifted_trace = [[coord[0] - min_x, coord[1] - min_y]
                         for coord in trace]
        shifted_trace_grp.append(shifted_trace)
    return shifted_trace_grp


def get_min_coords(trace_group):
    min_x_coords = []
    min_y_coords = []
    max_x_coords = []
    max_y_coords = []
    for trace in trace_group:
        x_coords = [coord[0] for coord in trace]
        y_coords = [coord[1] for coord in trace]
        min_x_coords.append(min(x_coords))
        min_y_coords.append(min(y_coords))
        max_x_coords.append(max(x_coords))
        max_y_coords.append(max(y_coords))
    return min(min_x_coords), min(min_y_coords), max(max_x_coords), max(max_y_coords)


def convert_to_img(trace_grps, box_size=100):
    patterns_enc = []
    classes_rejected = []
    for pattern in trace_grps:
        if "label" in pattern and pattern["label"] not in classes:
            classes.add(pattern["label"])
        # if pattern["label"] not in category:
        #     continue
        trace_group = pattern['traces']
        # mid coords needed to shift the pattern
        min_x, min_y, max_x, max_y = get_min_coords(trace_group)

        # traceGroup dimensions
        trace_grp_height, trace_grp_width = max_y - min_y, max_x - min_x

        # shift pattern to its relative position
        shifted_trace_grp = shift_trace_grp(
            trace_group, min_x=min_x, min_y=min_y)

        # Interpolates a pattern so that it fits into a box with specified size
        # method: LINEAR INTERPOLATION
        try:
            interpolated_trace_grp = interpolate(
                shifted_trace_grp, trace_grp_height=trace_grp_height, trace_grp_width=trace_grp_width, box_size=box_size - 1)
        except Exception as e:
            print(e)
            print('This data is corrupted - skipping.')
            classes_rejected.append(pattern.get('label'))
            continue

        # Get min, max coords once again in order to center scaled patter inside the box
        min_x, min_y, max_x, max_y = get_min_coords(interpolated_trace_grp)

        centered_trace_grp = center_pattern(
            interpolated_trace_grp, max_x=max_x, max_y=max_y, box_size=box_size)

        # Center scaled pattern so it fits a box with specified size
        pattern_drawn = draw_pattern(centered_trace_grp, box_size=box_size)
        # Make sure that patterns are thinned (1 pixel thick)
        pat_thinned = 1.0 - thin(1.0 - np.asarray(pattern_drawn))
        # plt.imshow(pat_thinned, cmap='gray')
        # plt.show()
        pattern_enc = {'features': pat_thinned, 'label': pattern.get('label')}

        patterns_enc.append(pattern_enc)
    return patterns_enc, classes_rejected


def process_inkml_files(inkml_files):
    data = []
    damaged_classes = []
    for file in tqdm(inkml_files):
        print(file)
        trace_groups = extract_trace_grps(file)
        patterns_enc, classes_rejected = convert_to_img(
            trace_groups, box_size=args["box_size"])
        damaged_classes.append(classes_rejected)
        data.append(patterns_enc)
    return data, damaged_classes


# args=dict()
# args["data_path"]="/home/jester/workspace/personal_projects/math_expression/dataset/"
# args["box_size"]=100
# args["version"]=["2011", "2012", '2013', "2014"]

train_inkml_files, test_inkml_files = extract_inkml_files()
print('Number of training INKML files:', len(train_inkml_files))
print('Number of testing INKML files:', len(test_inkml_files))

# collect all label classes
classes = set()
train, damaged_train = process_inkml_files(train_inkml_files)
test, damaged_test = process_inkml_files(test_inkml_files)
print(len(classes))

classes = sorted(classes)
print('Training set size:', len(train))
print('Testing set size:', len(test))
print('How many rejected trace groups:', len(damaged_train + damaged_test))

with open("dataset/classes.txt", "w") as file:
    file.write("\n".join(classes))
    print("Classes extracted and written to disk")

# # Dump extracted data
# outputs_dir = 'outputs'
# train_out_dir = os.path.join(outputs_dir, 'train')
# test_out_dir = os.path.join(outputs_dir, 'test')
# # Make directories if needed
# if not os.path.exists(outputs_dir):
#     os.mkdir(outputs_dir)
# if not os.path.exists(train_out_dir):
#     os.mkdir(train_out_dir)
# if not os.path.exists(test_out_dir):
#     os.mkdir(test_out_dir)

# with open(os.path.join(train_out_dir, 'train.pickle'), 'wb') as f:
#     pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)
#     print('Training data has been successfully dumped into', f.name)
# with open(os.path.join(test_out_dir, 'test.pickle'), 'wb') as f:
#     pickle.dump(test, f, protocol=pickle.HIGHEST_PROTOCOL)
#     print('Testing data has been successfully dumped into', f.name)
# # Save all labels in 'classes.txt' file
# with open('classes.txt', 'w') as f:
#     for r_class in classes:
#         f.write(r_class + '\n')
#     print('All classes that were extracted are listed in {} file.'.format(f.name))
