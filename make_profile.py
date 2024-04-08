import os
import cv2
import glob
import json
import warnings
import argparse
import numpy as np

# Collecting command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-r', '--rows', help='number of chessboard rows', type=int, required=True)
parser.add_argument('-c', '--columns', help='number of chessboard columns', type=int, required=True)
parser.add_argument(
    '-m', '--mode',
    help='how to obtain sample images needed for calibration. if set to IMAGE you need to '
         'provide a directory in --image-path that containing images to calibrate. if set to CAMERA it will open '
         'a the camera provided in --stream-url to capture images needed for calibration.',
    metavar='CALIBRATION_MODE', choices=['IMAGE', 'CAMERA'], required=True
)
parser.add_argument(
    '-u', '--stream-url', help='streaming url for a camera',
    metavar='SAVE_PATH', type=str, default=None
)
parser.add_argument(
    '-i', '--image-path', help='directory that contains images to make profile',
    metavar='SAVE_PATH', type=str, default=None
)
parser.add_argument(
    '-p', '--profile-name', help='name for the new profile',
    metavar='PROFILE_NAME', type=str, default='unnamed_profile'
)
parser.add_argument(
    '-s', '--save-path', help='directory to save the new camera profile',
    metavar='SAVE_PATH', type=str, default='profiles'
)
parser.add_argument(
    '--auto-confirm', help="don't ask for confirmation when reading images from a directory",
    action="store_true"
)
parser.add_argument(
    '-f', '--flip', help="flip image horizontally when in VIDEO mode (usually when using a camera)",
    action="store_true"
)
args = parser.parse_args()

# Checking arguments
if args.mode == 'IMAGE' and args.image_path is None:
    args.image_path = 'images/'
    warnings.warn(
        'You are using IMAGE mode and no image directory provided! Falling back to default folder (images/)'
    )

if args.mode == 'CAMERA' and args.stream_url is None:
    args.stream_url = 0
    warnings.warn(
        'You are using CAMERA mode and No camera stream url provided. Falling back to default camera'
    )


# Class to encode numpy arrays into json
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, objects):
        if isinstance(objects, np.integer):
            return int(objects)
        elif isinstance(objects, np.floating):
            return float(objects)
        elif isinstance(objects, np.ndarray):
            return objects.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(objects)


def find_chessboard(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray_image,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if ret is True:
        corners = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
    return ret, corners, img


def ask_confirmation():
    print('Please enter "Y" to confirm, or "N" to decline this image.')
    while True:
        key = cv2.waitKey(0)
        if key == ord('y') or key == ord('Y'):
            return True
        elif key == ord('n') or key == ord('N'):
            return False
        print('Please enter ether "Y" or "N" to confirm, or decline this image!')


# Defining needed variables
image = None
image_points = []
object_points = []
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
CHECKERBOARD = (args.rows, args.columns)
obj = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
obj[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Detect chessboard corners on input images
if args.mode == 'IMAGE':
    filenames = glob.glob('{}/*.jpg'.format(args.stream_url))
    for filename in filenames:
        image = cv2.imread(filename)
        ret, corners, image = find_chessboard(image)
        if ret is True:
            if not args.auto_confirm:
                cv2.imshow('Confirm? Y | N', image)
                confirmed = ask_confirmation()
                cv2.destroyWindow('Confirm? Y | N')
                if confirmed:
                    object_points.append(obj)
                    image_points.append(corners)
            else:
                object_points.append(obj)
                image_points.append(corners)
        else:
            print('No chessboard found!')

# Detect chessboard corners from a live camera
elif args.mode == 'CAMERA':
    if args.stream_url == '0':
        url = 0
    else:
        url = args.stream_url
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Error opening video stream!")
    while cap.isOpened():
        ret, frame = cap.read()
        if args.flip:
            frame = cv2.flip(frame, 1)
        cv2.imshow('Live Camera', frame)
        if cv2.waitKey(25) & 0xFF == ord(' '):
            ret, corners, image = find_chessboard(frame)
            if ret is True:
                cv2.imshow('Confirm? Y | N', image)
                confirmed = ask_confirmation()
                cv2.destroyWindow('Confirm? Y | N')
                if confirmed:
                    object_points.append(obj)
                    image_points.append(corners)
            else:
                print('No chessboard found!')
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()

cv2.destroyAllWindows()

# Check if any image exist
if image_points:
    h, w = image.shape[:2]
else:
    raise ValueError('No confirmed image found!')

# Calculating calibration parameters
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
    object_points, image_points, (w, h), None, None
)
new_matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w, h), 1, (w, h))
print('\nProfile created!')

# Calculate estimated calibration error
mean_error = 0
for i in range(len(object_points)):
    temp, _ = cv2.projectPoints(object_points[i], r_vecs[i], t_vecs[i], matrix, distortion)
    error = cv2.norm(image_points[i], temp, cv2.NORM_L2) / len(temp)
    mean_error += error
print('Estimated calibration error (smaller is better): {}'.format(mean_error / len(object_points)))

# Saving calibration profile as json
print('\nSaving {} calibration profile...'.format(args.profile_name))
save_path = os.path.join(args.save_path, '{}.json'.format(args.profile_name))
with open(save_path, "w") as profile:
    calibration_details = {
        'image_size': (w, h),
        'matrix': matrix,
        'new_matrix': new_matrix,
        'distortion': distortion,
        'r_vecs': r_vecs,
        't_vecs': t_vecs,
        'roi': roi
    }
    json.dump(calibration_details, profile, cls=NumpyArrayEncoder)
print('Done!')
