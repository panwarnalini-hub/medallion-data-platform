# Gesture classification


import time
import numpy as np
from collections import deque
from enum import Enum


class Gesture(Enum):
    NONE = "NONE"
    OPEN_PALM = "OPEN_PALM"
    FIST = "FIST"
    POINTING = "POINTING"
    THUMBS_UP = "THUMBS_UP"
    PEACE = "PEACE"
    SWIPE_LEFT = "SWIPE_LEFT"
    SWIPE_RIGHT = "SWIPE_RIGHT"


# Landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


class GestureClassifier:
    def __init__(self):
        self.wrist_history = deque(maxlen=15)
        self.time_history = deque(maxlen=15)
        self.last_swipe_time = 0
        self.current_gesture = Gesture.NONE
        self.hold_count = 0

        # Swipe thresholds - very sensitive
        self.SWIPE_MIN_DISTANCE = 0.05
        self.SWIPE_MIN_SPEED = 0.12
        self.SWIPE_COOLDOWN = 0.5
        
        # Static gesture thresholds
        self.BEND_THRESHOLD = 0.45
        self.HOLD_FRAMES = 6  # Require longer hold to avoid false triggers

    def _finger_bend(self, lm, mcp, pip, dip):
        """0 = straight, 1 = bent"""
        v1 = lm[mcp] - lm[pip]
        v2 = lm[dip] - lm[pip]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return 1.0 - (np.arccos(np.clip(cos_a, -1, 1)) / np.pi)

    def _get_bends(self, lm):
        return {
            "index": self._finger_bend(lm, INDEX_MCP, INDEX_PIP, INDEX_DIP),
            "middle": self._finger_bend(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP),
            "ring": self._finger_bend(lm, RING_MCP, RING_PIP, RING_DIP),
            "pinky": self._finger_bend(lm, PINKY_MCP, PINKY_PIP, PINKY_DIP),
        }

    def _is_thumb_extended(self, lm):
        """
        Check if thumb is extended outward (for THUMBS_UP).
        Works in any hand orientation ie thumb can point up, left, or right.
        
        Key insight: In THUMBS_UP, thumb tip is FAR from the other fingertips.
        In FIST, thumb is wrapped around and CLOSE to other fingers.
        """
        thumb_tip = lm[THUMB_TIP]
        thumb_mcp = lm[THUMB_MCP]
        index_tip = lm[INDEX_TIP]
        middle_tip = lm[MIDDLE_TIP]
        index_mcp = lm[INDEX_MCP]
        pinky_mcp = lm[PINKY_MCP]
        
        # Palm width for normalization
        palm_width = np.linalg.norm(index_mcp - pinky_mcp) + 1e-6
        
        # Check 1: Thumb tip distance from index fingertip
        # In FIST: thumb is close to curled fingers
        # In THUMBS_UP: thumb is far from curled fingers
        thumb_to_index_tip = np.linalg.norm(thumb_tip - index_tip) / palm_width
        thumb_to_middle_tip = np.linalg.norm(thumb_tip - middle_tip) / palm_width
        
        # Check 2: Thumb length (is thumb extended or curled?)
        thumb_length = np.linalg.norm(thumb_tip - thumb_mcp) / palm_width
        
        # Check 3: Thumb tip distance from palm center
        palm_center = (index_mcp + pinky_mcp) / 2
        thumb_from_palm = np.linalg.norm(thumb_tip - palm_center) / palm_width
        
        # THUMBS_UP: thumb is extended away from other fingers
        is_extended = (
            thumb_to_index_tip > 0.8 and      # Thumb far from index tip
            thumb_to_middle_tip > 0.8 and     # Thumb far from middle tip
            thumb_length > 0.4 and            # Thumb is extended
            thumb_from_palm > 0.6             # Thumb away from palm
        )
        
        return is_extended

    # Looks at recent movement direction and speed.
    def _detect_swipe(self):

        now = time.time()
        
        # Cooldown check
        if now - self.last_swipe_time < self.SWIPE_COOLDOWN:
            return Gesture.NONE
            
        if len(self.wrist_history) < 8:
            return Gesture.NONE

        # Get recent positions
        positions = list(self.wrist_history)
        times = list(self.time_history)
        
        # Check movement over last 8 frames
        start_x = positions[-8]
        end_x = positions[-1]
        dist = end_x - start_x
        
        elapsed = times[-1] - times[-8]
        if elapsed < 0.05:
            return Gesture.NONE
        
        speed = abs(dist) / elapsed
        
        # Check if movement is consistent (not jittery)
        # Count how many frames moved in same direction
        same_direction = 0
        for i in range(-7, 0):
            if dist > 0 and positions[i] < positions[i+1]:
                same_direction += 1
            elif dist < 0 and positions[i] > positions[i+1]:
                same_direction += 1
        
        # Need at least 4 frames moving in same direction
        consistent = same_direction >= 4
        
        if abs(dist) > self.SWIPE_MIN_DISTANCE and speed > self.SWIPE_MIN_SPEED and consistent:
            self.last_swipe_time = now
            self.wrist_history.clear()
            self.time_history.clear()
            self.hold_count = 0  # Reset hold to prevent static gesture after swipe
            return Gesture.SWIPE_RIGHT if dist > 0 else Gesture.SWIPE_LEFT

        return Gesture.NONE

    def _detect_static(self, lm, bends):
        t = self.BEND_THRESHOLD
        
        # Check finger states
        index_straight = bends["index"] < t
        middle_straight = bends["middle"] < t
        ring_straight = bends["ring"] < t
        pinky_straight = bends["pinky"] < t
        
        straight_count = sum([index_straight, middle_straight, ring_straight, pinky_straight])
        bent_count = 4 - straight_count
        
        # Check thumb
        thumb_extended = self._is_thumb_extended(lm)
        
        # OPEN_PALM: All 4 fingers straight
        if straight_count == 4:
            return Gesture.OPEN_PALM
        
        # THUMBS_UP: All 4 fingers bent and thumb extended outward
        if bent_count >= 3 and thumb_extended:
            return Gesture.THUMBS_UP
        
        # FIST: All 4 fingers bent and thumb NOT extended
        if bent_count >= 3 and not thumb_extended:
            return Gesture.FIST
        
        # POINTING: Only index straight
        if index_straight and not middle_straight and not ring_straight and not pinky_straight:
            return Gesture.POINTING
        
        # PEACE: Index and middle straight
        if index_straight and middle_straight and not ring_straight and not pinky_straight:
            return Gesture.PEACE

        return Gesture.NONE

    def classify(self, lm):
        now = time.time()
        self.wrist_history.append(lm[WRIST][0])
        self.time_history.append(now)

        bends = self._get_bends(lm)

        # ALWAYS check swipe first - takes priority
        swipe = self._detect_swipe()
        if swipe != Gesture.NONE:
            self.hold_count = 0
            self.current_gesture = Gesture.NONE
            return swipe, bends
        
        # Check if hand is moving
        if len(self.wrist_history) >= 5:
            recent_movement = abs(self.wrist_history[-1] - self.wrist_history[-5])
            if recent_movement > 0.03:  # Hand is moving
                return Gesture.NONE, bends

        # Static gesture detection
        static = self._detect_static(lm, bends)

        if static == self.current_gesture and static != Gesture.NONE:
            self.hold_count += 1
        else:
            self.current_gesture = static
            self.hold_count = 1

        if self.current_gesture == Gesture.NONE:
            return Gesture.NONE, bends

        # Require holding for several frames
        if self.hold_count >= self.HOLD_FRAMES:
            return self.current_gesture, bends

        return Gesture.NONE, bends
