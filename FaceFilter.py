# Alexandra French
# This code will add filters to a webcam or video feed based on input

import cv2
import numpy as np


class FaceFilters:
    # control viewing bounding boxes
    __show_face = False
    __show_left_eye = False
    __show_mouth = False

    # control viewing filters
    __mask_filter = False
    __monocle_filter = False
    __big_left_eye = False
    __big_mouth = False

    def run(self, video=None):
        # load the classifiers
        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        left_eye_classifier = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
        mouth_classifier = cv2.CascadeClassifier('haarcascade_smile.xml')

        # load all filter assets
        face_mask = cv2.imread('mask.png', cv2.IMREAD_UNCHANGED)
        monocle = cv2.imread('monocle.png', cv2.IMREAD_UNCHANGED)

        # open webcam if no video feed
        if video is None:
            video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # print out initial prompts
        self.print_help()

        # while the video feed is still open
        while video.isOpened():
            # get next frame and gray version
            _, frame = video.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # find faces and add filter
            faces = face_classifier.detectMultiScale(frame_gray, 1.5, 8)
            self.perform_img_filter(faces, (0, 255, 0), self.__show_face, self.__mask_filter, face_mask, frame)

            # find left eyes and add filter
            left_eyes = left_eye_classifier.detectMultiScale(frame_gray, 1.5, 10)
            self.perform_img_filter(left_eyes, (255, 0, 0), self.__show_left_eye, self.__monocle_filter, monocle, frame)
            self.perform_big_filter(left_eyes, (255, 0, 0), self.__show_left_eye, self.__big_left_eye, frame, 3)

            # find mouths and add filters
            mouths = mouth_classifier.detectMultiScale(frame_gray, 2.2, 27)
            self.perform_big_filter(mouths, (0, 0, 255), self.__show_mouth, self.__big_mouth, frame, 3)

            # show the resultant image with all filters applied
            cv2.imshow("Face Filters", frame)

            # check for user input
            key = cv2.waitKey(1) & 0xFF

            # quit
            if key == ord('q') or key == ord(" ") or key == 27:
                break

            # show help prints
            if key == ord('h'):
                self.print_help()

            # check filter and debug info
            self.toggle_clear(key)
            self.toggle_boxes(key)
            self.toggle_filters(key)

        # Close the camera and windows
        print("No more video feed detected. Goodbye.")
        video.release()
        cv2.destroyAllWindows()

    def perform_big_filter(self, detected, color, show_box, add_filter, frame, size):
        # for each detected, if wanted, show filter or bounding box
        for (x, y, w, h) in detected:
            x2, y2 = x + w, y + h

            if show_box:
                cv2.rectangle(frame, (x, y), (x2, y2), color, 3)

            if add_filter:
                # grab the detected frame from the image and give it an alpha channel
                new_img = frame[y:y2, x:x2]
                new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2RGBA)
                new_img[:, :, 3] = np.ones(new_img.shape[1]) * 245
                new_img = cv2.resize(new_img, (w, h), interpolation=cv2.INTER_AREA)

                # for each pixel in the detected frame, if around the edge, set alpha to 0
                for i in range(h):
                    for j in range(w):
                        if i in range(h // 7, h // 7 * 6) and j in range(w // 7, w // 7 * 6):
                            continue
                        new_img[i][j][3] = 0

                # blur the opacity of the 0 alphas with the full alphas, making a blurred mask
                img_blurred = cv2.GaussianBlur(new_img, (5, 5), cv2.BORDER_DEFAULT)

                # resize detected img and blur mask to bg and setup bg for new sizes
                resize_w = w // 7 * size
                resize_h = h // 7 * size
                new_img = cv2.resize(new_img, (w + resize_w, h + resize_h), interpolation=cv2.INTER_AREA)
                img_blurred = cv2.resize(img_blurred, (w + resize_w, h + resize_h), interpolation=cv2.INTER_AREA)
                background = frame[y - resize_h // 2:y2 + resize_h // 2, x - resize_w // 2:x2 + resize_w // 2]

                # add bigger image to frame where original was
                self.add_image(background, new_img, img_blurred)

    def perform_img_filter(self, detected, color, show_box, add_filter, filter_img, frame):
        # for each detected, if wanted, show filter or bounding box
        for (x, y, h, w) in detected:
            x2, y2 = x + w, y + h

            if show_box:
                cv2.rectangle(frame, (x, y), (x2, y2), color, 3)

            if add_filter:
                # resize the filter img to the detected and get the background for the filter img
                resized_img = cv2.resize(filter_img, (w, h),
                                         interpolation=cv2.INTER_AREA)
                background = frame[y:y2 + 5, x:x2]

                # add the filter img on the filter img background
                self.add_image(background, resized_img)

    def add_image(self, bg, img_filter, mask=None):
        # determine shape from image and background
        filter_h, filter_w = img_filter.shape[:2]
        bg_height, bg_width = bg.shape[:2]

        # determine whether to mix filter with a mask version or not
        if mask is None:
            mask = img_filter

        # for every pixel add the image w/ its opacity to the background
        for i in range(filter_h):
            for j in range(filter_w):
                if i >= bg_height or j >= bg_width:
                    continue
                opacity = mask[i][j][3] / 255.0
                bg[i][j] = opacity * img_filter[i][j][:3] \
                           + (1 - opacity) * bg[i][j]

    def toggle_clear(self, key):
        # Clear bounding boxes
        if key == ord('z'):
            print("Cleared all bounding boxes.")
            self.__show_face = False
            self.__show_mouth = False
            self.__show_left_eye = False

        # Clear filters
        if key == ord('x'):
            print("Cleared all filters.")
            self.__mask_filter = False
            self.__monocle_filter = False
            self.__big_mouth = False

        # Clear all filters and bounding boxes
        if key == ord('c'):
            print("Cleared everything.")
            self.__show_face = False
            self.__show_mouth = False
            self.__show_left_eye = False
            self.__mask_filter = False
            self.__monocle_filter = False
            self.__big_mouth = False

    def toggle_boxes(self, key):
        # enable boxes bounding boxes based on user input
        if key == ord('1'):
            if self.__show_face:
                self.__show_face = False
            else:
                self.__show_face = True
        if key == ord('2'):
            if self.__show_left_eye:
                self.__show_left_eye = False
            else:
                self.__show_left_eye = True
        if key == ord('3'):
            if self.__show_mouth:
                self.__show_mouth = False
            else:
                self.__show_mouth = True

    def toggle_filters(self, key):
        # enable filters based on user input
        if key == ord('a'):
            if self.__mask_filter:
                self.__mask_filter = False
            else:
                self.__mask_filter = True
        if key == ord('s'):
            if self.__monocle_filter:
                self.__monocle_filter = False
            else:
                self.__monocle_filter = True
        if key == ord('d'):
            if self.__big_left_eye:
                self.__big_left_eye = False
            else:
                self.__big_left_eye = True
        if key == ord('f'):
            if self.__big_mouth:
                self.__big_mouth = False
            else:
                #print("Smile!")
                self.__big_mouth = True


    def print_help(self):
        print("To view each bounding box, press one of the following:")
        print("1: Face\t2: Left Eye\t3: Mouth\t")
        print("To view each filter, press one of the following:")
        print("a: Mask\ts: Monocle\td: Big Mouth\tf: Big left eye")
        print("z: Clear boxes\tx: Clear filters\tc: Clear everything")


if __name__ == "__main__":
    faceFilter = FaceFilters()
    faceFilter.run()
