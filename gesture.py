"""Hand + pose landmark detection helpers."""

import cv2
import mediapipe as mp

_mp_hands = mp.solutions.hands
_mp_pose  = mp.solutions.pose

# Finger tip / pip / mcp landmark indices
_TIPS = [
    _mp_hands.HandLandmark.INDEX_FINGER_TIP,
    _mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    _mp_hands.HandLandmark.RING_FINGER_TIP,
    _mp_hands.HandLandmark.PINKY_TIP,
]
_PIPS = [
    _mp_hands.HandLandmark.INDEX_FINGER_PIP,
    _mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
    _mp_hands.HandLandmark.RING_FINGER_PIP,
    _mp_hands.HandLandmark.PINKY_PIP,
]
_MCPS = [
    _mp_hands.HandLandmark.INDEX_FINGER_MCP,
    _mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
    _mp_hands.HandLandmark.RING_FINGER_MCP,
    _mp_hands.HandLandmark.PINKY_MCP,
]

_PL = _mp_pose.PoseLandmark


class GestureDetector:
    def __init__(self):
        self._hands = _mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self._pose = _mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0,   # fastest model
        )

    # ------------------------------------------------------------------
    def detect(self, frame_bgr) -> dict:
        """
        Returns a dict with:
          hand_raised   bool
          fist_clenched bool
          open_palm     bool
          wrist         (x, y) px or None
          left_shoulder (x, y) px or None
          right_shoulder(x, y) px or None
          torso_center  (x, y) px or None
          chest_center  (x, y) px or None  # between shoulders and torso
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        hand_res = self._hands.process(rgb)
        pose_res  = self._pose.process(rgb)

        result = {
            'hand_raised':    False,
            'fist_clenched':  False,
            'open_palm':      False,
            'wrist':          None,
            'left_shoulder':  None,
            'right_shoulder': None,
            'torso_center':   None,
            'chest_center':   None,
        }

        # ---- Pose -------------------------------------------------------
        shoulder_y_px = None
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark

            ls = lm[_PL.LEFT_SHOULDER]
            rs = lm[_PL.RIGHT_SHOULDER]
            lh = lm[_PL.LEFT_HIP]
            rh = lm[_PL.RIGHT_HIP]

            ls_px = (int(ls.x * w), int(ls.y * h))
            rs_px = (int(rs.x * w), int(rs.y * h))
            lh_px = (int(lh.x * w), int(lh.y * h))
            rh_px = (int(rh.x * w), int(rh.y * h))

            result['left_shoulder']  = ls_px
            result['right_shoulder'] = rs_px

            torso_x = (ls_px[0] + rs_px[0] + lh_px[0] + rh_px[0]) // 4
            torso_y = (ls_px[1] + rs_px[1] + lh_px[1] + rh_px[1]) // 4
            result['torso_center'] = (torso_x, torso_y)

            shoulder_avg_y = (ls_px[1] + rs_px[1]) // 2
            chest_y = (shoulder_avg_y + torso_y) // 2
            result['chest_center'] = (torso_x, chest_y)

            shoulder_y_px = min(ls.y, rs.y) * h

        # ---- Hand -------------------------------------------------------
        if hand_res.multi_hand_landmarks:
            hl = hand_res.multi_hand_landmarks[0].landmark

            wrist = hl[_mp_hands.HandLandmark.WRIST]
            result['wrist'] = (int(wrist.x * w), int(wrist.y * h))

            # Hand raise: wrist pixel-y < shoulder pixel-y (y grows downward)
            if shoulder_y_px is not None:
                result['hand_raised'] = (wrist.y * h) < shoulder_y_px

            # Fist: all 4 finger tips below their PIP joints
            result['fist_clenched'] = all(
                hl[tip].y > hl[pip].y
                for tip, pip in zip(_TIPS, _PIPS)
            )

            # Open palm: all 4 finger tips above their MCP joints
            result['open_palm'] = all(
                hl[tip].y < hl[mcp].y
                for tip, mcp in zip(_TIPS, _MCPS)
            )

        return result

    def close(self):
        self._hands.close()
        self._pose.close()
