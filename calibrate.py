################ 
import cv2
import json
import argparse
import numpy as np

# Collecting command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-p', '--profile', help='path to profile to use for calibration',
    metavar='PROFILE_NAME', type=str, required=True
)
parser.add_argument(
    '-u', '--url',
    help='path to the input data. could be either a path to single image, a video file, a live video url.'
         'or 0 to use your webcam or usb camera.',
    metavar='INPUT_PATH', type=str, required=True
)
parser.add_argument(
    '-m', '--mode', help='the mode which calibration should operate on.',
    metavar='CALIBRATION_MODE', choices=['IMAGE', 'VIDEO'], required=True
)
parser.add_argument(
    '-s', '--save',
    help='name to save the results. if provided it will save the results.',
    metavar='SAVE_LOCATION', default=None, type=str, required=False
)
parser.add_argument(
    '--no-output', help='do not display calibration results (Only works in IMAGE mode).',
    action='store_true'
)
parser.add_argument(
    '-f', '--flip', help="flip image horizontally when in VIDEO mode (usually when using a camera)",
    action="store_true"
)
args = parser.parse_args()

# Checking arguments
if args.no_output and args.save is None:
    raise ValueError('a save location must be provided when --no-output flag is used')

# Opening profile
with open(args.profile) as json_file:
    profile = json.load(json_file)

matrix = np.asarray(profile['matrix'])
new_matrix = np.asarray(profile['new_matrix'])
distortion = np.asarray(profile['distortion'])
image_size = profile['image_size']
x, y, w, h = profile['roi']

# Calibrating camera on image mode
if args.mode.upper() == 'IMAGE':
    image = cv2.imread(args.url)
    result = cv2.undistort(image, matrix, distortion, None, new_matrix)
    result = result[y:y + h, x:x + w]
    if not args.no_output:
        cv2.imshow('Result', result)
        cv2.waitKey(0)
    if args.save is not None:
        print('Saving calibration results...')
        cv2.imwrite('output/{}.jpg'.format(args.save), result)
        print('Done!')


# Calibrating camera on video mode
elif args.mode.upper() == 'VIDEO':
    map_x, map_y = cv2.initUndistortRectifyMap(matrix, distortion, None, new_matrix, image_size, 5)
    if len(args.url) == 1:
        args.url = int(args.url)
    cap = cv2.VideoCapture(args.url)
    if not cap.isOpened():
        print("Error opening video stream or file!")
    if args.save is not None:
        save = True
        writer = cv2.VideoWriter('output/{}.avi'.format(args.save), cv2.VideoWriter_fourcc(*'MJPG'), 10, (w, h))
        print('Start calibrating the stream...')
    else:
        save = False
    print('Press Q to stop calibrating')
    while cap.isOpened():
        ret, frame = cap.read()
        if args.flip:
            frame = cv2.flip(frame, 1)
        result = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        result = result[y:y + h, x:x + w]
        if not args.no_output:
            cv2.imshow('Result', result)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        if save:
            writer.write(result)
    cap.release()
    writer.release()
    print('Done!')

cv2.destroyAllWindows()
