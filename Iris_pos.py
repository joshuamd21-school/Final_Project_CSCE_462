import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui as mouse
import math
import time


class eye_tracking():
    def __init__(self):
        self.camera = cv.VideoCapture(0)  # setting up of camera
        self.screen_width, self.screen_height = mouse.size()
        self.MOUSESPEED = 20
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )  # face mesh used for eye and iris detection

        self.calibrating = False
        self.CALIBRATION_STEPS = 30  # amount of data collected for calibration
        self.calibration_state = "left"
        self.calibrated = False
        self.calibration_horiz = []
        self.calibration_vert = []
        self.cal_point_left = 0
        self.cal_point_right = 0
        self.cal_point_top = 0
        self.cal_point_bottom = 0

        self.vert_slope = -0.15
        self.vert_intercept = 3.33

        self.horiz_slope = 0.16
        self.horiz_intercept = 3.19

        self.V_ITER_MAX = 20
        self.v_iter = 0
        self.vertical_ratios = np.zeros(self.V_ITER_MAX)

        self.H_ITER_MAX = 20
        self.h_iter = 0
        self.horizontal_ratios = np.zeros(self.H_ITER_MAX)

        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
                         249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 154,
                          133, 173, 157, 158, 159, 160, 161, 246]

        self.RIGHT_IRIS = [474, 475, 476, 477]
        self.LEFT_IRIS = [469, 470, 471, 472]

        self.L_LEFT = [33]
        self.L_RIGHT = [133]

        self.R_LEFT = [362]  # right eye left point
        self.R_TOP = [443]  # right eye top point
        self.R_BOTTOM = [450]  # right eye bottom point
        self.R_RIGHT = [263]  # right eye right point

    def iris_ratio(self, iris_center, point1, point2):
        center_to_point1_dist = euclidean_distance(iris_center, point1)
        total_distance = euclidean_distance(point1, point2)
        ratio = center_to_point1_dist/total_distance
        return ratio

    def move_mouse(self, horiz_ratio, vert_ratio):

        horiz_ratio, vert_ratio = self.smoothen(horiz_ratio, vert_ratio)

        mouse.moveTo(self.screen_width*(1/self.horiz_slope * horiz_ratio - self.horiz_intercept), mouse.position().y
                     #  self.screen_height/2
                     #  self.screen_height*(1/self.vert_slope * vert_ratio - self.vert_intercept)
                     )
        if vert_ratio <= 0.35:
            mouse.move(0, self.MOUSESPEED)
        elif vert_ratio >= 0.65:
            mouse.move(0, -self.MOUSESPEED)

    def compute_equations(self):
        print("left ratio: ", self.cal_point_left)
        print("right ratio: ", self.cal_point_right)
        print("top ratio: ", self.cal_point_top)
        print("bottom ratio: ", self.cal_point_bottom)
        self.vert_slope = self.cal_point_bottom - self.cal_point_top
        self.horiz_slope = self.cal_point_right - self.cal_point_left

        self.vert_intercept = self.cal_point_top/self.vert_slope
        self.horiz_intercept = self.cal_point_left/self.horiz_slope

    def calibrate(self, horiz_ratio, vert_ratio):
        state = self.calibration_state
        if state == "done":
            self.compute_equations()
            self.calibrated = True

        if state == "left" or state == "right":
            self.calibration_horiz.append(horiz_ratio)
            if len(self.calibration_horiz) >= self.CALIBRATION_STEPS:
                if state == "left":
                    self.cal_point_left = np.mean(self.calibration_horiz)
                else:
                    self.cal_point_right = np.mean(self.calibration_horiz)
                self.calibration_horiz.clear()
                self.switch_calibration_state()
        elif state == "top" or state == "bottom":
            self.calibration_vert.append(vert_ratio)
            if len(self.calibration_vert) >= self.CALIBRATION_STEPS:
                if state == "top":
                    self.cal_point_top = np.mean(self.calibration_vert)
                else:
                    self.cal_point_bottom = np.mean(self.calibration_vert)
                self.calibration_horiz.clear()
                self.switch_calibration_state()

    def switch_calibration_state(self):
        self.calibrating = False
        if self.calibration_state == "left":
            self.calibration_state = "right"
        elif self.calibration_state == "right":
            self.calibration_state = "top"
        elif self.calibration_state == "top":
            self.calibration_state = "bottom"
        elif self.calibration_state == "bottom":
            self.calibration_state = "done"

    def smoothen(self, horiz_ratio, vert_ratio):

        self.horizontal_ratios[self.h_iter] = horiz_ratio
        self.h_iter += 1
        if (self.h_iter >= self.H_ITER_MAX):
            self.h_iter = 0
        self.vertical_ratios[self.v_iter] = vert_ratio\

        self.v_iter += 1
        if (self.v_iter >= self.V_ITER_MAX):
            self.v_iter = 0

        smooth_ratio_horiz = 0.0
        for ratio in self.horizontal_ratios:
            smooth_ratio_horiz += ratio
        smooth_ratio_horiz /= len(self.horizontal_ratios)

        smooth_ratio_vert = 0.0
        for ratio in self.vertical_ratios:
            smooth_ratio_vert += ratio
        smooth_ratio_vert /= len(self.vertical_ratios)

        return smooth_ratio_horiz, smooth_ratio_vert

    def track(self):

        ret, frame = self.camera.read()

        frame = cv.flip(frame, 1)

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        img_h, img_w = frame.shape[:2]
        if results.multi_face_landmarks:

            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(
                int) for p in results.multi_face_landmarks[0].landmark])

            center_left, center_right = self.locate_irises(
                frame, mesh_points, circle_irises=True)

            points = [mesh_points[tracking.R_RIGHT][0], mesh_points[tracking.R_LEFT]
                      [0], mesh_points[tracking.R_BOTTOM][0], mesh_points[tracking.R_TOP][0]]
            self.circle_points(frame, points)

            horiz_ratio = tracking.iris_ratio(
                center_right, mesh_points[tracking.R_RIGHT], mesh_points[tracking.R_LEFT][0])
            vert_ratio = tracking.iris_ratio(
                center_right, mesh_points[tracking.R_TOP], mesh_points[tracking.R_BOTTOM][0])

            self.display_ratios(frame, horiz_ratio, vert_ratio)
            if not self.calibrated:
                if not self.calibrating:
                    cv.putText(frame, "Please look at the " +
                               self.calibration_state + " of the sreen then press c key", (30, 120), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)
                    if cv.waitKey(1) & 0xFF == ord('c'):
                        print("calibrating")
                        self.calibrating = True
                else:
                    cv.putText(frame, "calibrating", (30, 120),
                               cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)
                if self.calibrating:
                    self.calibrate(horiz_ratio, vert_ratio)
            else:
                self.move_mouse(horiz_ratio, vert_ratio)

            cv.imshow('img', frame)

    def circle_points(self, frame, points):
        for point in points:
            cv.circle(frame, point, 2,
                      (255, 255, 255), 1, cv.LINE_AA)

    def display_ratios(self, frame, horiz_ratio, vert_ratio):
        cv.putText(
            frame, f"Horizontal ratio: {horiz_ratio: 0.2f}", (30, 30), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)
        cv.putText(
            frame, f"Vertical ratio: {vert_ratio: 0.2f}", (30, 90), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)

    def locate_irises(self, frame, mesh_points, circle_irises=False):

        (l_cx, l_cy), l_radius = cv.minEnclosingCircle(
            mesh_points[tracking.LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv.minEnclosingCircle(
            mesh_points[tracking.RIGHT_IRIS])

        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
        if circle_irises:
            cv.circle(frame, center_left, int(l_radius),
                      (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius),
                      (255, 0, 255), 1, cv.LINE_AA)
        return center_left, center_right


def euclidean_distance(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2 - x1)**2 + (y2-y1)**2)
    return distance


if __name__ == "__main__":

    mouse.FAILSAFE = False
    tracking = eye_tracking()

    while True:
        tracking.track()
        # Exit the loop if the 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    tracking.camera.release()
    cv.destroyAllWindows()
